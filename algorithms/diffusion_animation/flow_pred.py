import pytorch_lightning as pl
from omegaconf import DictConfig
import torch
import torchvision

import wandb

import random

import numpy as np
from .denoising_diffusion import Unet
from .warp import warp
from .augmentation import Augmentor

from utils.video_prediction.visualization import log_video

class Autoencoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model_enc = Unet(
            64,
            channels=3,
            out_dim=cfg.latent_dim,
            dim_mults=(1, 2, 4),
            time_in=False
        )
        self.model_dec = Unet(
            64,
            channels=cfg.latent_dim + 3,
            dim_mults=(1, 2, 4),
            out_dim=3,
            time_in=False
        )
        #print('old version without clamping')

    def forward(self, x, flow, return_latent=False):
        l_ = self.model_enc(2 * x - 1.0)
        l_ = torch.clamp(l_, -1.0, 1.0) # clamp
        l = warp(l_, None, flow, mode='forward')
        x = self.model_dec(torch.cat((l, 2 * x - 1), dim=1))
        x = torch.clamp(x, -1.0, 1.0)

        if not return_latent:
            return (x + 1.0) / 2.0
        else:
            return l

    def encode(self, x):
        return torch.clamp(self.model_enc(2 * x - 1.0), -1.0, 1.0) # clamp
        #return self.model_enc(2 * x - 1.0)

    def decode(self, latent, x):
        x = self.model_dec(torch.cat((latent, 2 * x - 1), dim=1))
        x = torch.clamp(x, -1.0, 1.0)

        return (x + 1.0) / 2.0

class FlowPred(pl.LightningModule):
    """
    Our proposed method of diffusion_animation filtering.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        imsz = [int(x) for x in cfg.image_size.split(",")]
        self.image_w, self.image_h = imsz[0], imsz[1]

        self.augmentor = Augmentor()
        self.ae = Autoencoder(cfg)

    def configure_optimizers(self):
        self.optimizers = torch.optim.Adam(self.ae.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        return self.optimizers

    def training_step(self, batch, batch_idx):
        batch = self.augmentor(batch)
        img, tgt, flow = batch
        flow = flow + torch.randn(flow.shape, device=flow.device)

        if random.random() > self.cfg.ae_frac:
            out = self.ae(img, flow)
            loss = torch.nn.functional.mse_loss(out, tgt)
        else:
            out = self.ae(img, torch.zeros_like(flow))
            loss = torch.nn.functional.mse_loss(out, img)

        self.log_dict({
            "train/loss": loss
        })

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            img, tgt, flow = batch
            bsz = batch[0].shape[0]

            out = self.ae(img, flow)
            loss = torch.nn.functional.mse_loss(out, tgt)

            self.log_dict({
                "val/loss": loss
            })

            # prepare for wandb upload
            def chunk(x):
                x[:, 0, 0, 0] = x[:, 0, 0, 0] * 0.95                 # not completely white so wandb doesn't bug out
                return list(torch.chunk(x, bsz))

            # Flows
            gt_flow = torchvision.utils.flow_to_image(batch[2]) / 255.0

            self.logger.log_image(key='original', images=chunk(batch[0]), step=self.global_step)
            self.logger.log_image(key='target', images=chunk(batch[1]), step=self.global_step)

            self.logger.log_image(key='gt_flow', images=chunk(gt_flow), step=self.global_step)
            self.logger.log_image(key='target_p', images=chunk(out), step=self.global_step)

            #log_video(batch[0], batch[1], out, key="compare")
