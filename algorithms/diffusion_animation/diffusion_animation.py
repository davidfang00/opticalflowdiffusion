import pytorch_lightning as pl
from omegaconf import DictConfig
import torch
import torchvision

from .denoising_diffusion import Unet, ConditionalDiffusion
from utils.image_prediction.logging import log_photos
from utils.video_prediction.visualization import log_video

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * torch.norm(input - target, dim=1))


class FrameGenerator(pl.LightningModule):
    """
    Our proposed method of diffusion_animation filtering.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.automatic_optimization = False
        self.cfg = cfg
        self.image_size = cfg.image_size

        self._model = Unet(
            64,  # number of dimensions within the Unet itself
            channels=3 + 3 + 2,  # input dimension
            out_dim=3  # output dimension
        )
        self.diffusion_model = ConditionalDiffusion(
            self._model,
            self.image_size,
            objective="pred_noise"
        )

    def configure_optimizers(self):
        self.optimizers = [
            torch.optim.Adam(self.diffusion_model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        ]

        return self.optimizers, []

    def training_step(self, batch, batch_idx):
        target = batch[:, :3, :, :]
        cond = batch[:, 3:, :, :]

        optimizer_dynamics = self.optimizers[0]
        self.toggle_optimizer(optimizer_dynamics)

        loss = self.diffusion_model(target, cond)
        optimizer_dynamics.zero_grad()
        self.manual_backward(loss)
        optimizer_dynamics.step()
        self.untoggle_optimizer(optimizer_dynamics)

        self.log_dict({
            "train/loss": loss
        })
        self.log_grad_norm_stat()

    def validation_step(self, batch, batch_idx):
        val_length = batch.size[1]

        batch_ = batch[:, 0]
        target = batch_[:, :3, :, :]
        cond = batch_[:, 3:, :, :]

        last_frames = cond[:, :3, :, :]
        flows = cond[:, 3:, :, :]

        with torch.no_grad():
            loss = self.diffusion_model(target, cond)
            samples = self.diffusion_model.sample(batch_size=batch_.shape[0], external_cond=cond)

            self.log_dict({
                "val/loss": loss
            })

            for photos, key in zip((samples, target, last_frames, flows), ("samples", "targets", "last_frames", "flows")):
                photos = photos.reshape((-1, val_length, 3, self.image_size, self.image_size))[:, 0]
                log_photos((photos, ), self, keyword=f"val/{key}")

        if batch_idx % 1 == 0:
            samples = []
            for iter in range(val_length):
                cond = batch[:, iter, 3:, :, :]

                if iter != 0:
                    cond[:, :3, :, :] = samples[-1][:, :3, :, :] # replace with previous generation

                with torch.no_grad():
                    samples.append(self.diffusion_model.sample(batch_size=batch_.shape[0], external_cond=cond))

            log_video(
                torch.stack(samples, dim=0), batch[:, :, :3, :, :].transpose(0, 1),
                step=self.global_step,
                namespace='val',
                context_frames=1,
                logger=self.logger.experiment
            )


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


class FlowCompleter(pl.LightningModule):
    """
    Our proposed method of diffusion_animation filtering.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.automatic_optimization = False
        self.cfg = cfg
        self.image_size = cfg.image_size

        self.model = Unet(
            64,  # number of dimensions within the Unet itself
            channels=3 + 2,  # input dimension
            out_dim=2,  # output dimension
            time_in=False
        )
        self.null_embedding = [
            torch.nn.Parameter(torch.ones(1)),
            torch.nn.Parameter(torch.ones(1))
        ] # learnable

        self.lmbd = 0.2

    def configure_optimizers(self):
        self.optimizers = [
            torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay),
            torch.optim.Adam(self.null_embedding, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        ]

        return self.optimizers, []

    def _sparse_from_dense(self, dense_flow):
        sparse_flow = torch.cat([
            torch.full((dense_flow.shape[0], self.image_size, self.image_size), self.null_embedding[idx],
                                 device=self.device) for idx in [0, 1]
        ], dim=1)

        flow_mags = torch.norm(dense_flow, dim=1)

        smoother = torch.mean(flow_mags)
        for frame in range(dense_flow.shape[0]): # pick some random flows
            picked = torch.utils.data.WeightedRandomSampler(torch.flatten(flow_mags[frame]) + smoother,
                                                            torch.randint(8, (1,)).item() + 1, replacement=False)
            picked = torch.tensor(list(picked))
            picked = torch.stack((picked // self.image_size, picked % self.image_size), dim=1)
            sparse_flow[frame, 3:5, picked[:, 0], picked[:, 1]] = dense_flow[frame, 3:5, picked[:, 0], picked[:, 1]]

        sparse_flow.requires_grad_()
        return sparse_flow, flow_mags

    def _flow_mse_loss(self, input, target, flow_mags):
        return weighted_mse_loss(
            input,
            target,
            self.lmbd + flow_mags / torch.amax(flow_mags, dim=(1, 2), keepdim=True)
        )

    def training_step(self, batch, batch_idx):
        dense_flow = batch[:, -2:, :, :]
        frame = batch[:, 3:6, :, :]

        sparse_flow, flow_mags = self._sparse_from_dense(dense_flow)

        optimizer_dynamics = self.optimizers[0]
        self.toggle_optimizer(optimizer_dynamics)

        out = self.model(torch.cat((sparse_flow, frame), dim=1))
        loss = self._flow_mse_loss(out, dense_flow, flow_mags)

        optimizer_dynamics.zero_grad()
        self.manual_backward(loss)
        optimizer_dynamics.step()
        self.untoggle_optimizer(optimizer_dynamics)

        self.log_dict({
            "train/loss": loss
        })
        self.log_grad_norm_stat()

    def validation_step(self, batch, batch_idx):
        batch = batch[:, 0]
        dense_flow = batch[:, -2:, :, :]
        frame = batch[:, 3:6, :, :]

        sparse_flow, flow_mags = self._sparse_from_dense(dense_flow)

        with torch.no_grad():
            out = self.model(torch.cat((sparse_flow, frame), dim=1))
            loss = self._flow_mse_loss(out, dense_flow, flow_mags)

            if batch_idx % 1 == 0:
                self.log_dict({
                    "val/loss": loss
                })
                log_photos((frame, ), self, keyword="frames")
                log_photos((torchvision.utils.flow_to_image(dense_flow), ), self, keyword="real_flows")
                log_photos((torchvision.utils.flow_to_image(out), ), self, keyword="predictions")

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