import pytorch_lightning as pl
from omegaconf import DictConfig
import torch
import torchvision
from .augmentation import Augmentor

from .pwc_net import PWCNet
from .losses import total_loss

class PWCLearner(pl.LightningModule):
    """
    Our proposed method of diffusion_animation filtering.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.augmentor = Augmentor()
        self.model = PWCNet()

    def configure_optimizers(self):
        self.optimizers = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        return self.optimizers

    def chunk(self, x):
        bsz = x.shape[0]
        x[:, 0, 0, 0] = x[:, 0, 0, 0] * 0.95  # not completely white so wandb doesn't bug out
        return list(torch.chunk(x, bsz))

    def loss(self, flow_fwd, flow_bwd, occ, warped_imgs, tar_ds):
        assert len(flow_fwd) == len(flow_bwd) and len(flow_fwd) == len(occ) and len(flow_fwd) == len(warped_imgs), "list length mismatch"

        loss = 0
        # level_weights = [0.005, 0.01, 0.02, 0.08, 0.32]
        level_weights = [0.005, 0.01, 0.02, 0.08, 0.32]
        # level_weights = [1.5, 1.5, 2, 2, 3]

        for i in range(len(flow_fwd)):
            ref = tar_ds[i]
            future_warped = warped_imgs[i][0]
            past_warped = warped_imgs[i][1]
            f_flow = flow_fwd[i]
            p_flow = flow_bwd[i]
            occlusions = occ[i]
            level_loss = total_loss(ref, past_warped, future_warped, p_flow, f_flow, occlusions)

            loss += level_loss * level_weights[i]

        return loss

    def training_step(self, batch, batch_idx):
        frame1, frame2, frame3, gt_flow = batch
        flow_fwd, flow_bwd, occ, warped_imgs, tar_ds = self.model(frame2, [frame1, frame3])

        loss = self.loss(flow_fwd, flow_bwd, occ, warped_imgs, tar_ds)

        fullres_fwd_flow = flow_fwd[0]
        self.log_dict({
            "train/loss": loss,
            "train/flow_fwd_min": torch.min(fullres_fwd_flow),
            "train/flow_fwd_max": torch.max(fullres_fwd_flow),
            "train/flow_fwd_mean": torch.mean(fullres_fwd_flow),
            "train/flow_fwd_std": torch.mean(torch.std(fullres_fwd_flow, dim=0))
        })

        self.log('loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frame1, frame2, frame3, gt_flow = batch
        # print(frame2.mean())

        with torch.no_grad():
            # loss = self.loss(tgt_, cond, flow_)
            flow_fwd, flow_bwd, occ, warped_imgs, tar_ds = self.model(frame2, [frame1, frame3])
            loss = self.loss(flow_fwd, flow_bwd, occ, warped_imgs, tar_ds)

            # mse = torch.nn.functional.mse_loss(samples, tgt)
            # flow_mse = torch.nn.functional.mse_loss(flow_, p_flows / self.flow_max)

            self.log_dict({
                "val/loss": loss,
            }, sync_dist=True)

            fwd_flows = torchvision.utils.flow_to_image(flow_fwd[0]) / 255 * torch.max(frame2)
            bwd_flows = torchvision.utils.flow_to_image(flow_bwd[0]) / 255 * torch.max(frame2)
            gt_flows = torchvision.utils.flow_to_image(gt_flow) / 255 * torch.max(frame2)

            occlusions = occ[0][:, 0: None]
            warped_imgs_fwd = warped_imgs[0][0]
            warped_imgs_bwd = warped_imgs[0][1]
            target = tar_ds[0]

            # print(warped_imgs_fwd.mean())
            print(flow_fwd[0].max(), flow_fwd[0].min(), gt_flow.max(), gt_flow.min())

            reconstructed_img = occ[0][:, 0, None] * warped_imgs_fwd + occ[0][:, 1, None] * warped_imgs_bwd

            combined_frames = torch.cat((frame1, frame2, frame3), dim=3)
            fwd_flow = torch.cat((frame2, frame3, fwd_flows), dim=3)
            bwd_flow = torch.cat((frame1, frame2, bwd_flows), dim=3)
            fwd_warped = torch.cat((frame2, frame3, warped_imgs_fwd), dim=3)
            bwd_warped = torch.cat((frame2, frame1, warped_imgs_bwd), dim=3)
            gt_fwd = torch.cat((gt_flows, fwd_flows), dim=3)
            orig_reconstructed = torch.cat((frame2, reconstructed_img), dim=3)

            self.logger.log_image(key='combined_frames', images=self.chunk(combined_frames), step=self.global_step)
            self.logger.log_image(key='fwd_flow', images=self.chunk(fwd_flow), step=self.global_step)
            self.logger.log_image(key='bwd_flow', images=self.chunk(bwd_flow), step=self.global_step)
            self.logger.log_image(key='occlusions', images=self.chunk(occlusions), step=self.global_step)
            self.logger.log_image(key='fwd_warped', images=self.chunk(fwd_warped), step=self.global_step)
            self.logger.log_image(key='bwd_warped', images=self.chunk(bwd_warped), step=self.global_step)
            self.logger.log_image(key='target', images=self.chunk(target), step=self.global_step)
            self.logger.log_image(key='gt_fwd_flow', images=self.chunk(gt_fwd), step=self.global_step)
            self.logger.log_image(key='reconstructed_comb', images=self.chunk(orig_reconstructed), step=self.global_step)

            


