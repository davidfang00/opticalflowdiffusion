import pytorch_lightning as pl
from omegaconf import DictConfig
import torch
import torchvision

from .denoising_diffusion import Unet, ConditionalDiffusion
from .flow_pred import FlowPred, Autoencoder

from .warp import warp
from .augmentation import Augmentor

from utils.video_prediction.visualization import log_video
from utils.wandb_utils import download_latest_checkpoint

import random
from pathlib import Path
from omegaconf import OmegaConf
import os

class UnetWithWarp(torch.nn.Module):
    def __init__(self, cfg, unet, full_output, nan_safe=True):
        super().__init__()
        self.cfg = cfg
        self.flow_max = cfg.flow_max
        self.dim = cfg.latent_dim if cfg.latent else 3

        self.model = unet
        self.full_output = full_output
        self.nan_safe = nan_safe

        if cfg.zero_init:
            self.model.final_conv.weight.data = torch.zeros_like(self.model.final_conv.weight.data)
            self.model.final_conv.bias.data = torch.zeros_like(self.model.final_conv.bias.data)

    def _warp(self, image, flow, **kwargs):
        return warp(image[:, :self.dim], None, flow * self.flow_max, mode='forward', **kwargs)

    def forward(self, x, external_cond = None, t=None, self_cond=None, additional_out = False):
        if self.nan_safe:
            x = x.clone()
            where_nans = torch.isnan(x)
            x[where_nans] = 0.0
            where_nans = torch.any(where_nans, dim=1)[:, None]

            flow = self.model(torch.cat((x, where_nans), dim=1), external_cond, t, self_cond)
        else:
            flow = self.model(x, external_cond, t, self_cond)

        if external_cond is not None:
            warped = self._warp(external_cond, flow[:, :2])
        else:
            warped = self._warp(x[:, :self.dim], flow[:, :2])
        #warped = warp(external_cond[:, :self.dim], None, flow * self.flow_max, mode='forward')
        #warped = warp(None, external_cond, flow * self.flow_max, mode='backward')[0]

        out = warped
        if self.full_output:
            out = torch.cat((out, flow), dim=1)

        if not additional_out:
            return out
        else:
            return torch.cat((out, flow), dim=1)

class FlowDiffuser(pl.LightningModule):
    """
    Our proposed method of diffusion_animation filtering.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.flow_max = cfg.flow_max
        self.latent_max = cfg.latent_max
        self.is_diffusion = cfg.is_diffusion
        self.latent = cfg.latent
        self.target = cfg.target

        self.augmentor = Augmentor()
        if self.latent:
            self.ae = Autoencoder(cfg)

            expected_path = "outputs/loaded_checkpoints/diffusion_control/" + cfg.ae + "/model.ckpt"
            if not os.path.exists(expected_path):
                model_path = download_latest_checkpoint("diffusion_control/" + cfg.ae, Path("outputs/loaded_checkpoints"))
            else:
                model_path = expected_path
            #model_path = download_latest_checkpoint("diffusion_control/" + cfg.ae, Path("outputs/loaded_checkpoints"))
            state_dict = torch.load(model_path)["state_dict"]
            state_dict = {k.replace("ae.", ""): v for k, v in state_dict.items() if k.startswith("ae.")}
            self.ae.load_state_dict(state_dict)
            for p in self.ae.parameters():
                p.requires_grad = False
            print("ONCE YOU FIX THE CLAMPING FOR LATENTS THEN COME BACK AND REMOVE LATENT MAX")


        self.dim = cfg.latent_dim if self.latent else 3
        if self.target == "target":
            unet_dims = self.dim + 1 # 1 for nan
        elif self.target == "joint":
            unet_dims = self.dim + 3
        else:
            unet_dims = 2

        self.unet = Unet(
            64,  # number of dimensions within the Unet itself
            channels=self.dim + unet_dims * int(self.is_diffusion),  # input dimension
            out_dim=2,
            time_in=cfg.is_diffusion
        )
        if cfg.target in ["target", "joint"]:
            self._model = UnetWithWarp(cfg, self.unet, full_output=cfg.target == "joint")
        else:
            self._model = self.unet

        if self.is_diffusion:
            self.model = ConditionalDiffusion(
                self._model,
                cfg.image_size,
                objective="pred_x0",
                channels=cfg.latent_dim if cfg.latent else (2 + 1 * int(cfg.target == "target") + 3 * int(cfg.target == "joint")),
                auto_normalize=False,
                noise_space = "image" if cfg.noiser == "image" else "flow",
                timesteps = cfg.timesteps,
                min_snr_loss_weight=True
            )
        else:
            self.model = self._model

    def configure_optimizers(self):
        self.optimizers = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        return self.optimizers

    def preprocess(self, batch, aug=True):
        if aug:
            batch = self.augmentor(batch)

        img, tgt, flow = batch
        flow = torch.clamp(flow / self.flow_max, -1.0, 1.0)

        if self.latent:
            with torch.no_grad():
                img = self.ae.encode(img) / self.latent_max
                tgt = self.ae.encode(tgt) / self.latent_max
                img = torch.clamp(img, -1.0, 1.0)
                tgt = torch.clamp(tgt, -1.0, 1.0)
        else:
            img = 2 * img - 1.0
            tgt = 2 * tgt - 1.0

        ret = []

        if self.target == 'target':
            #ret.append(tgt)
            ret.append(warp(img, None, flow * self.flow_max, mode='forward'))
            print('using warped input instead of true target')
        elif self.target == 'joint':
            ret.append(torch.cat((warp(img, None, flow * self.flow_max, mode='forward'), flow), dim=1))
            print('using warped input instead of true target')
        else:
            ret.append(flow)

        ret.append(img)
        ret.append(flow)

        return tuple(ret)

    def loss(self, tgt, cond, flow, override=None):
        if self.is_diffusion:
            if self.cfg.target == "target":
                loss = self.model(tgt, external_cond=cond, additional_tgt=flow, additional_weight=self.cfg.flow_weight, model_out_override=override)
            else:
                loss = self.model(tgt, external_cond=cond, model_out_override=override)
        else:
            if override is not None:
                out = self.model(cond, additional_out = self.cfg.target == "target")
            else:
                out = override
            if self.cfg.target in ["target", "joint"]:
                loss = torch.nn.functional.mse_loss(out[:, :self.dim], tgt)
                loss += self.cfg.flow_weight * torch.nn.functional.mse_loss(out[:, self.dim:], flow)
            else:
                loss = torch.nn.functional.mse_loss(out, flow)

        return loss

    def sample(self, cond, flow):
        bsz = flow.shape[0]

        if self.is_diffusion:
            if self.cfg.target == "target":
                samples, flow = self.model.sample(batch_size=bsz, external_cond=cond, additional_tgt=flow, return_all_timesteps=True)
            elif self.cfg.target == "joint":
                joint_samples = self.model.sample(batch_size=bsz, external_cond=cond, return_all_timesteps=True)
                samples = joint_samples[:, :, :self.dim]
                flow = joint_samples[:, :, self.dim:]
            else:
                flow = self.model.sample(batch_size=bsz, external_cond=cond, return_all_timesteps=True)
                img = cond[:, :self.dim]
                samples = warp(img, None, flow[:, -1], mode='forward')
        else:
            if self.cfg.target in ["target", "joint"]:
                if self.cfg.target == "target":
                    samples = self.model(cond, True)
                else:
                    samples = self.model(cond)
                flow = samples[:, -2:]
                samples = samples[:, :self.dim]
            elif self.cfg.target == "flow":
                flow = self.model(cond)
                samples = warp(cond[:, :self.dim], None, flow, mode='forward')

        return samples, flow


    def training_step(self, batch, batch_idx):
        batch = self.preprocess(batch)
        loss = self.loss(*batch)
        tgt, cond, flow = batch

        self.log_dict({
            "train/loss": loss,
            "train/cond_min": torch.min(cond),
            "train/cond_max": torch.max(cond),
            "train/cond_mean": torch.mean(cond),
            "train/cond_std": torch.mean(torch.std(cond, dim=0)),
            "train/flow_min": torch.min(flow),
            "train/flow_max": torch.max(flow),
            "train/flow_mean": torch.mean(flow),
            "train/flow_std": torch.mean(torch.std(flow, dim=0))
        })

        return loss

    def validation_step(self, batch, batch_idx):
        img, tgt, flow = batch
        tgt_, cond, flow_ = self.preprocess(batch, aug=False)
        bsz = img.shape[0]

        with torch.no_grad():
            loss = self.loss(tgt_, cond, flow_)
            samples, p_flows = self.sample(cond, flow_)
            if self.is_diffusion:
                mid_samples = samples[:, ::50]
                samples = samples[:, -1]
                if self.target == "target":
                    p_flows = [None] + [p * self.flow_max for p in p_flows[1:]]
                    mid_flows = p_flows[1::50]
                    p_flows = p_flows[-1]
                elif self.target == "joint":
                    mid_flows = p_flows[:, ::50] * self.flow_max
                    p_flows = p_flows[:, -1] * self.flow_max
            mse = torch.nn.functional.mse_loss(samples, tgt if not self.latent else self.ae.encode(tgt))
            if self.target == "target":
                ideal_loss = self.loss(tgt_, cond, flow_, override=(warp(cond[:, :self.dim], None, flow_ * self.flow_max, mode='forward'), flow_))
            elif self.target == "joint":
                ideal_loss = self.loss(tgt_, cond, flow_, override=(torch.cat((warp(cond[:, :self.dim], None, flow_ * self.flow_max, mode='forward'), flow_), dim=1), None))

            self.log_dict({
                "val/loss": loss,
                "val/mse": mse,
                "val/cond_min": torch.min(cond),
                "val/cond_max": torch.max(cond),
                "val/cond_mean": torch.mean(cond),
                "val/cond_std": torch.mean(torch.std(cond, dim=0)),
                "val/flow_min": torch.min(flow),
                "val/flow_max": torch.max(flow),
                "val/flow_mean": torch.mean(flow),
                "val/flow_std": torch.mean(torch.std(flow, dim=0)),
                "val/samples_min": torch.min(samples),
                "val/samples_max": torch.max(samples),
                "val/samples_mean": torch.mean(samples),
                "val/samples_std": torch.mean(torch.std(samples, dim=0)),
                "val/p_flow_min": torch.min(p_flows),
                "val/p_flow_max": torch.max(p_flows),
                "val/p_flow_mean": torch.mean(p_flows),
                "val/p_flow_std": torch.mean(torch.std(p_flows, dim=0)),
                "val/ideal_loss": ideal_loss
            }, sync_dist=True)

            def chunk(x):
                x[:, 0, 0, 0] = x[:, 0, 0, 0] * 0.95  # not completely white so wandb doesn't bug out
                return list(torch.chunk(x, bsz))

            # Flows
            flos = torchvision.utils.flow_to_image(torch.cat((flow, p_flows, flow - p_flows), dim=0)) / 255.0
            gt_flow = flos[:bsz]
            sample_flow = flos[bsz:2 * bsz]
            diff_flow = flos[2 * bsz:]

            self.logger.log_image(key='original', images=chunk(img), step=self.global_step)
            self.logger.log_image(key='target', images=chunk(tgt), step=self.global_step)
            self.logger.log_image(key='diffusion_tgt', images=chunk((tgt_[:, :self.dim] + 1.0) * 0.5), step=self.global_step)
            if not self.latent:
                self.logger.log_image(key='original_warped', images=chunk(warp(img, None, flow, mode='forward')), step=self.global_step)

            self.logger.log_image(key='gt_flow', images=chunk(gt_flow), step=self.global_step)
            self.logger.log_image(key='target_p', images=chunk(sample_flow), step=self.global_step)
            self.logger.log_image(key='concat', images=chunk(torch.cat((gt_flow, sample_flow), dim=3)), step=self.global_step)
            self.logger.log_image(key='difference', images=chunk(diff_flow), step=self.global_step)

            if self.latent:
                dec = self.ae.decode(samples * self.latent_max, img)
                self.logger.log_image(key='samples', images=chunk(dec), step=self.global_step)
                #log_video(img, dec, key="compare")
                self.logger.log_image(key='compare', images=chunk(torch.cat((img, dec), dim=-1)), step=self.global_step)
                dec_gt = self.ae(img, flow)
                self.logger.log_image(key='dec_gt', images=chunk(dec_gt), step=self.global_step)
            else:
                self.logger.log_image(key='samples', images=chunk(samples), step=self.global_step)

            if self.is_diffusion and self.target in ["target", "joint"]:
                if self.latent:
                    mid_samples_ = []
                    for mid_sample in torch.chunk(mid_samples, mid_samples.shape[1], dim=1):
                        mid_samples_.append(self.ae.decode(mid_sample[:, 0] * self.latent_max, img))
                    mid_samples = torch.cat(mid_samples_, dim=-1)
                else:
                    mid_samples = torch.cat(torch.chunk(mid_samples, mid_samples.shape[1], dim=1), dim=-1)[:, 0]
                    mid_samples = torch.clamp(mid_samples, -1.0, 1.0)

                if self.target == "target":
                    mid_flows = [torchvision.utils.flow_to_image(mid_flow) / 255.0 for mid_flow in mid_flows]
                    mid_flows = torch.cat(mid_flows, dim=-1)
                elif self.target == "joint":
                    mid_flows_orig_shape = list(mid_flows.shape)
                    mid_flows = torch.reshape(mid_flows, (-1, 2, mid_flows.shape[-2], mid_flows.shape[-1]))
                    mid_flows = torchvision.utils.flow_to_image(mid_flows) / 255.0
                    mid_flows_orig_shape[2] = 3
                    mid_flows = torch.reshape(mid_flows, mid_flows_orig_shape)

                    mid_flows = torch.cat(torch.chunk(mid_flows, mid_flows.shape[1], dim=1), dim=-1)[:, 0]
                    mid_flows = torch.clamp(mid_flows, -1.0, 1.0)
                self.logger.log_image(key='mid_samples', images=chunk(mid_samples), step=self.global_step)
                self.logger.log_image(key='mid_flows', images=chunk(mid_flows), step=self.global_step)
            #log_video(gt_flow, sample_flow, key="compare")

            if self.is_diffusion and self.target in ["target", "joint"]:
                last_step = self.model.model(tgt_, cond, torch.zeros((bsz, ), device=self.device, dtype=torch.long), None, additional_out=True)
                last_step = last_step[:, -2:]
                self.log_dict({
                    "val/last_step": torch.nn.functional.mse_loss(last_step, flow_)
                }, sync_dist=True)
                flos = torchvision.utils.flow_to_image(torch.cat((flow_, last_step), dim=0)) / 255.0
                gt_flow2 = flos[:bsz]
                last_step_flow = flos[bsz:]
                self.logger.log_image(key='last_step', images=chunk(torch.cat((gt_flow2, last_step_flow), dim=-1)), step=self.global_step)

        if self.is_diffusion and self.target in ["target", "joint"]:
            with torch.set_grad_enabled(True):
                p_flows.requires_grad_(True)
                #warped_p_flows = warp(cond, None, p_flows, mode='forward')
                #loss = torch.nn.functional.mse_loss(warped_p_flows, tgt_[:, :self.dim])
                loss = self.model._loss(warp(cond, None, p_flows, mode='forward'), tgt_[:, :self.dim], None, flow_, cond, p_flows / self.flow_max, 0.0)
                loss.backward()
                grad_flow = -p_flows.grad.clone() # negative so this reflects the drcn of grad descent
                p_flows.grad = None
                p_flows.requires_grad_(False)

                grad_flow = torchvision.utils.flow_to_image(grad_flow) / 255.0
                grad_flow = list(torch.chunk(grad_flow, bsz, dim=0))
                self.logger.log_image(key='grad_flow', images=grad_flow, step=self.global_step)

                
    def log_grad_norm_stat(self):
        with torch.no_grad():
            grad_norms = []
            gpr = []  # gradient-to-parameter ratio
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad_norms.append(torch.norm(param.grad).item())
                    gpr.append(torch.norm(param.grad) / torch.norm(param))
            grad_norms = torch.tensor(grad_norms)
            gpr = torch.tensor(gpr)
            self.log_dict({
                "train/grad_norm/min": grad_norms.min(),
                "train/grad_norm/max": grad_norms.max(),
                "train/grad_norm/std": grad_norms.std(),
                "train/grad_norm/mean": grad_norms.mean(),
                "train/grad_norm/median": torch.median(grad_norms),
                "train/gpr/min": gpr.min(),
                "train/gpr/max": gpr.max(),
                "train/gpr/std": gpr.std(),
                "train/gpr/mean": gpr.mean(),
                "train/gpr/median": torch.median(gpr),
            })
