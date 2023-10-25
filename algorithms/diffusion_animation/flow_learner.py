import pytorch_lightning as pl
from omegaconf import DictConfig
import torch
import torchvision

from .denoising_diffusion import Unet, ConditionalDiffusion
from .flow_pred import FlowPred, Autoencoder
from .filter import ConvToFilter

from .warp import warp, filter_to_flow, invert_filter, nan_mse, fill_holes_nan, nan_charbonnier, edgeaware_smoothness1
from .augmentation import Augmentor
from .flow_diffuser import UnetWithWarp
from .softsplat_new import softsplat as softsplat_new

from utils.video_prediction.visualization import log_video
from utils.wandb_utils import download_latest_checkpoint

import random, math
from pathlib import Path
from omegaconf import OmegaConf
import os, time

class FilterUnet(torch.nn.Module):
    def __init__(self, radius, c2f=False):
        super().__init__()
        self.radius = radius
        self.dim = 81
        if c2f:
            self.unet = Unet(
                64,
                channels=6,
                out_dim=2 if self.radius == None else self.dim + 4,
                time_in=False
            )
            self.c2f = ConvToFilter(self.radius, self.dim)
        else:
            self.unet = Unet(
                64,
                channels=6,
                out_dim=2 if self.radius == None else self.radius ** 2 + 4,
                time_in=False
            )
            self.c2f = None

    def forward(self, *args):
        if self.c2f is not None:
            out = self.unet(*args)
            cols = out[:, -4:]

            out_ = self.c2f(out[:, :-4])
            out = torch.cat((out_, cols), dim=1)    
        else:
            out = self.unet(*args) 

        mean_val = [self.radius ** 2 + 1] * (self.radius ** 2 + 4)
        mean_val = torch.Tensor(mean_val).to(out.device)
        mean_val[-4:-1] = 2.0

        return (out + 1.0) / mean_val[None, :, None, None]


class FlowLearner(pl.LightningModule):
    """
    Our proposed method of diffusion_animation filtering.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        if 'radius' in dir(cfg):
            self.radius = cfg.radius
            if 'flow_max' in dir(cfg):
                raise ValueError('cannot specify both flow_max and radius')
            self.flow_max = self.radius // 2
            self.rep = 'filter'
        else:
            self.radius = None
            self.flow_max = cfg.flow_max
            self.rep = 'flow'

        self.augmentor = Augmentor()
        if self.rep == 'flow':
            self.unet = UnetWithWarp(cfg, Unet(
                    64,
                    channels=6,
                    out_dim=3, #3 for optical flow + weights map
                    time_in=False
                ), False, nan_safe=False)
        elif self.rep == 'filter':
            self.unet = FilterUnet(self.radius, c2f=self.cfg.c2f)
        self.model = self.unet

    def configure_optimizers(self):
        self.optimizers = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        return self.optimizers

    def chunk(self, x):
        bsz = x.shape[0]
        x[:, 0, 0, 0] = x[:, 0, 0, 0] * 0.95  # not completely white so wandb doesn't bug out
        return list(torch.chunk(x, bsz))

    def preprocess(self, batch, aug=True):
        if aug:
            batch = self.augmentor(batch)

        img, tgt, flow = batch
        flow = torch.clamp(flow / self.flow_max, -1.0, 1.0)

        img = 2 * img - 1.0
        tgt = 2 * tgt - 1.0

        ret = []

        ret.append(tgt)
        ret.append(torch.cat((img, tgt), dim=1))
        ret.append(flow)

        return tuple(ret)

    def masked_mse_loss(self, out, target, mask):
        mask = mask.to(torch.float)
        loss = torch.nn.functional.mse_loss(out, target, reduction='none')
        loss = loss * mask

        loss = loss.sum() / mask.sum() 
        return loss

    def index_from_filter(self, fil, flow):
        pass

    def loss(self, tgt, cond, flow_, override_flow=None):
        #out = self.model(cond)
        #warped = warp(cond[:, :3], None, out, mode='forward', rep=self.rep)
        if override_flow is None:
            out = self.model(cond, additional_out = True)
            # flow_pred = out[:, -2:]

            flow_weight_pred = out[:, -3:]
            flow_pred = flow_weight_pred[:, :2] * self.flow_max
            warp_weights = flow_weight_pred[:, 2:]
        else:
            flow_pred = override_flow * self.flow_max
            warp_weights = torch.ones(flow_pred.shape)[:, :1].to(self.device)

        if self.rep == 'filter':
            out_noim = out.clone()
            out_noim[:, -1] = torch.zeros_like(out_noim[:, -1])
            warped_noim = warp(cond[:, :3], None, out_noim, mode='forward', rep=self.rep)

            out = filter_to_flow(out) / self.flow_max

        loss = 0.0

        #flow_loss = 0.0 * torch.nn.functional.mse_loss(out, flow_)
        #loss = photo_loss = #torch.nn.functional.mse_loss(warped, cond[:, 3:])
        # warped_gt = self.model._warp(cond, flow_)
        photo_loss = []
        #levels = [1, 2, 4, 8, 16]
        levels = list(range(1, 17))
        levels = [1, 2, 4, 5, 7, 8, 10, 11, 14, 16]
        #levels = [1]
        #levels = [16]

        input_img = cond[:, :3, :, :]

        for level in levels:
            photo_loss_lvl = []

            # downsampled_input = downsample_interpolation(input_img, scale = 1/level)
            # downsampled_flow = downsample_interpolation(flow_pred, scale = 1/level) / level # rescale the flow so it is within image
            # downsampled_weights = downsample_interpolation(warp_weights, scale = 1/level)
            # warped_input_with_weights = softsplat(downsampled_input, downsampled_flow, downsampled_weights, "soft")
            # warped = warped_input_with_weights[:, :-1, :, :]
            # weights = warped_input_with_weights[:, -1:, :, :]
            # filled_input = fill_holes_nan(warped, weights)

            # downsampled_tgt = downsample_interpolation(tgt, scale = 1/level)
            # photo_loss_lvl.append(nan_mse(downsampled_tgt, filled_input))

            for a in range(level):
                for b in range(level):
                    warped_input_with_weights = softsplat_new(input_img, flow_pred, warp_weights, strMode= "soft", scale=level, offset=[a, b])
                    warped = warped_input_with_weights[:, :-1, :, :]
                    weights = warped_input_with_weights[:, -1:, :, :]
                    filled_input = fill_holes_nan(warped, weights)

                    downsampled_tgt_with_weights = softsplat_new(tgt, torch.zeros_like(flow_), torch.ones_like(warp_weights), strMode= "soft", scale=level, offset=[a, b])
                    downsampled_tgt = downsampled_tgt_with_weights[:, :-1, :, :]
                    # level_loss = nan_mse(downsampled_tgt, filled_input)
                    level_loss = nan_charbonnier(downsampled_tgt, filled_input)
                    photo_loss_lvl.append(level_loss)

                    # photo_loss_lvl.append(nan_mse(self.model._warp(warped_gt, torch.zeros_like(flow_), scale=level, offset=[a, b]), self.model._warp(cond, flow_pred, scale=level, set_nans=False, offset=[a, b])))
            
            #photo_loss.append(nan_mse(self.model._warp(warped_gt, torch.zeros_like(flow_), scale=level), self.model._warp(cond, flow_pred, scale=level, set_nans=False)))
            #if level > 1:
            #    photo_loss.append(nan_mse(self.model._warp(warped_gt, torch.zeros_like(flow_), scale=level, get_variance=True), self.model._warp(cond, flow_pred, scale=level, set_nans=False, get_variance=True)) * (level ** 2))

            #    torchvision.utils.save_image(self.model._warp(cond, flow_pred, scale=level, set_nans=False, get_variance=True), f'levels_var2_{level}.png')
            #torch.abs(self.model._warp(warped_gt, torch.zeros_like(flow_), scale=level, set_nans=False, get_variance=True) - self.model._warp(cond, flow_pred, scale=level, set_nans=False, get_variance=True))
            #torchvision.utils.save_image(torch.abs(self.model._warp(warped_gt, torch.zeros_like(flow_), scale=level, set_nans=False) - self.model._warp(cond, flow_pred, scale=level, set_nans=False)), f'levels_var2_{level}.png')
            photo_loss.append(sum(photo_loss_lvl) / len(photo_loss_lvl))
        #loss = max(*photo_loss) 
        loss = sum(photo_loss) / len(photo_loss)
        smoothness_loss = edgeaware_smoothness1(input_img, flow_pred)
        loss += smoothness_loss * 0.01

        if self.rep == 'filter':
            if self.cfg.occlusion_mask:
                inverted = invert_filter(out_noim)
                mask = torch.gt(torch.sum(inverted[:, :self.radius ** 2], dim=1), 0.25)
            else:
                mask = torch.full_like(cond[:, 3], True)
            mask = mask.to(out.device)[:, None]
            noim_photo_loss = self.masked_mse_loss(warped_noim, cond[:, 3:], mask)
            
            sparsity_loss = torch.nn.functional.l1_loss(out[:, :self.radius ** 2], torch.zeros_like(out[:, :self.radius ** 2]))
            loss += noim_photo_loss + sparsity_loss * self.cfg.sparsity_weight

        return loss

    def sample(self, cond, flo, log_additional=False):
        bsz = flo.shape[0]

        if self.rep == 'flow':
            out = self.model(cond, additional_out = True) 
            # flow = out[:, -2:] * self.flow_max
            # samples = out[:, :-2]

            flow_weight_pred = out[:, -3:]
            flow = flow_weight_pred[:, :2] * self.flow_max
            warp_weights = flow_weight_pred[:, 2:]

            samples_with_weights = softsplat_new(cond[:, :3], flow, warp_weights, "soft", scale = 1, offset = [0,0])
            samples = samples_with_weights[:, :-1, :, :]
            sample_weights = samples_with_weights[:, -1:, :, :]
            samples = fill_holes_nan(samples, sample_weights)
            return samples, flow, warp_weights
        else:
            flow = self.model(cond)
            samples = warp(cond[:, :3], None, flow, mode='forward', rep=self.rep)

        if self.rep == 'filter':
            if log_additional:
                flow2 = flow.clone()
                flow2[:, -1] = torch.zeros_like(flow2[:, -1])
                no_col_samples = warp(cond[:, :3], None, flow2, mode='forward', rep=self.rep)
                self.logger.log_image(key='no_col', images=self.chunk(no_col_samples), step=self.global_step)
                self.logger.log_image(key='col_weighted', images=self.chunk(flow[:, None, -1] * flow[:, -4:-1]), step=self.global_step)

                flow_ = torch.cat((flow[:, :self.radius ** 2], flow[:, -1, None]), dim=1) # remove the color columns
                fsum = torch.sum(flow_, dim=1, keepdim=True)
                fsum_ = torch.ones((bsz, 1, fsum.shape[-2] + 2, fsum.shape[-1] + 2))
                fsum_[:, :, 1:-1, 1:-1] = fsum
                
                fmin = torch.min(flow_, dim=1, keepdim=True)[0]
                fmax = torch.max(flow_, dim=1, keepdim=True)[0]
                fcol = flow_[:, -1, None]
                fsparse = torch.max(torch.abs(flow_), dim=1, keepdim=True)[0] / (1e-4 + torch.sum(torch.abs(flow_), dim=1, keepdim=True))
                self.log_dict({
                    "val/filter_sum": torch.mean(fsum),
                    "val/filter_min": torch.min(fmin),
                    "val/filter_max": torch.max(fmax),
                    "val/filter_col": torch.mean(fcol),
                    "val/filter_sparsity": torch.mean(fsparse)
                }, sync_dist=True)

                scale = 8
                scaled_flow = flow[2, :, ::scale, ::scale] * 10 
                self.logger.log_image(key='filters', images=self.chunk(self.filter_to_image(scaled_flow, src=0.5 * (cond[2, :3, ::scale, ::scale]) + 1.0)[None, ...]), step=self.global_step)
                inverted_scaled_flow = invert_filter(scaled_flow[None, ...])[0]
                self.logger.log_image(key='i_filters', images=self.chunk(self.filter_to_image(inverted_scaled_flow, src=0.5 * (cond[2, -3:, ::scale, ::scale]) + 1.0)[None, ...]), step=self.global_step)

                self.logger.log_image(key='filter_sum_', images=self.chunk(fsum_), step=self.global_step)
                self.logger.log_image(key='filter_min_', images=self.chunk(fmin), step=self.global_step)
                self.logger.log_image(key='filter_max_', images=self.chunk(fmax), step=self.global_step)
                self.logger.log_image(key='filter_col_', images=self.chunk(fcol), step=self.global_step)
                self.logger.log_image(key='filter_sparsity_', images=self.chunk(fsparse), step=self.global_step)
                self.logger.log_image(key='filter_colors', images=self.chunk(flow_[:, -4:-1]), step=self.global_step)

            flow = filter_to_flow(flow)

        return samples, flow

    def training_step(self, batch, batch_idx):
        # img, tgt, flow = batch
        tgt, cond, flow = self.preprocess(batch, aug=self.cfg.train_aug)
        loss = self.loss(tgt, cond, flow)

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

        self.log('loss', loss, prog_bar=True)
        #self.logger.log_image(key='train_original', images=self.chunk((cond[:, :3] + 1.0) / 2.0), step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        img, tgt, flow = batch
        tgt_, cond, flow_ = self.preprocess(batch, aug=False)
        bsz = img.shape[0]

        with torch.no_grad():
            loss = self.loss(tgt_, cond, flow_)
            ideal_loss = self.loss(tgt_, cond, flow_, override_flow=flow_)
            samples, p_flows, warp_weights = self.sample(cond, flow_, log_additional=True)
            samples = torch.where(torch.isnan(samples), 0.0, samples)
            mse = torch.nn.functional.mse_loss(samples, tgt)
            flow_mse = torch.nn.functional.mse_loss(flow_, p_flows / self.flow_max)

            self.log_dict({
                "val/loss": loss,
                "val/ideal_loss": ideal_loss,
                "val/mse": mse,
                "val/flow_mse": flow_mse,
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
                "val/p_flow_std": torch.mean(torch.std(p_flows, dim=0))
            }, sync_dist=True)

            # Flows
            flos = torchvision.utils.flow_to_image(torch.cat((flow, p_flows, flow - p_flows), dim=0)) / 255.0
            gt_flow = flos[:bsz]
            sample_flow = flos[bsz:2 * bsz]
            diff_flow = flos[2 * bsz:]

            self.logger.log_image(key='original', images=self.chunk(img), step=self.global_step)
            self.logger.log_image(key='target', images=self.chunk(tgt), step=self.global_step)
            self.logger.log_image(key='original_warped', images=self.chunk(warp(img, None, flow, mode='forward', warp_style="avg", set_nans=False)), step=self.global_step)
            self.logger.log_image(key='flow_warp', images=self.chunk(warp(img, None, p_flows, mode='forward', rep=self.rep, set_nans=False, warp_style="avg").clamp(min=0, max=1)), step=self.global_step)
            self.logger.log_image(key='flow_var_warp', images=self.chunk(warp(img, None, p_flows, mode='forward', rep=self.rep, set_nans=False, scale=1, get_variance=True)), step=self.global_step)

            self.logger.log_image(key='gt_flow', images=self.chunk(gt_flow), step=self.global_step)
            self.logger.log_image(key='target_p', images=self.chunk(sample_flow), step=self.global_step)
            self.logger.log_image(key='concat', images=self.chunk(torch.cat((gt_flow, sample_flow), dim=3)), step=self.global_step)
            self.logger.log_image(key='difference', images=self.chunk(diff_flow), step=self.global_step)
            self.logger.log_image(key='warp_weights', images=self.chunk(warp_weights), step=self.global_step)

            self.logger.log_image(key='samples', images=self.chunk(samples), step=self.global_step)

            with torch.set_grad_enabled(True):
                p_flows.requires_grad_(True)
                loss = self.loss(tgt_, cond, flow_, override_flow=p_flows / self.flow_max)
                loss.backward()
                grad_flow = -p_flows.grad.clone() # negative so this reflects the drcn of grad descent
                p_flows.grad = None
                p_flows.requires_grad_(False)

                grad_flow = torchvision.utils.flow_to_image(grad_flow) / 255.0
                grad_flow = list(torch.chunk(grad_flow, bsz, dim=0))
                self.logger.log_image(key='grad_flow', images=grad_flow, step=self.global_step)

            #log_video(gt_flow, sample_flow, key="compare")



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
    
    def filter_to_image(self, filters, src=None):
        cws = None
        if filters.shape[0] > self.radius ** 2:
            cws = filters[-1]

        filters = filters[:self.radius ** 2]
        filters = filters.reshape((self.radius, self.radius, -1))
        if cws != None:
            filters[-1, -1] = cws.reshape((-1,))
        filters = filters.permute(2, 0, 1)
        filters = filters.reshape((-1, 1, self.radius, self.radius)).tile(1, 3, 1, 1)  # batch and rgb
        filters = torch.clamp(filters, 0.0, 1.0)
        filters[:, 2, self.radius // 2, :] = 0.33 * (1 + 2 * filters[:, 2, self.radius // 2, :])  # tint centers
        filters[:, 1, :, self.radius // 2] = 0.33 * (1 + 2 * filters[:, 1, :, self.radius // 2])

        if src != None:
            filters_ = src.permute(1, 2, 0).reshape((-1, 3, 1, 1)).repeat((1, 1, self.radius * 2, self.radius * 2))
            filters_[:, :, self.radius - self.radius // 2:self.radius + self.radius // 2 + 1, self.radius - self.radius // 2:self.radius + self.radius // 2 + 1] = filters
            filters = filters_
        filters = filters.repeat_interleave(3, dim=2).repeat_interleave(3, dim=3)  # scale up
        ret = torchvision.utils.make_grid(filters, nrow=round(math.sqrt(filters.shape[0])), pad_value=1.0)
        return ret
