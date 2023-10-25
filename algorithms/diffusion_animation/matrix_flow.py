import pytorch_lightning as pl
from omegaconf import DictConfig
import torch
import torchvision

import wandb

import numpy as np
import math

from .denoising_diffusion import Unet
from algorithms.diffusion_animation.future.raft import RAFT


class MatrixFlow(pl.LightningModule):
    """
    Our proposed method of diffusion_animation filtering.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        #self.automatic_optimization = False
        self.cfg = cfg
        imsz = [int(x) for x in cfg.image_size.split(",")]
        self.image_w, self.image_h = imsz[0], imsz[1]
        self.radius = cfg.radius
        assert(self.radius % 2 == 1)

        if "cols" in dir(cfg):
            if cfg.cols == "any":
                self.has = ["cols", "colweights"]
            else:
                self.has = ["colweights"]
        else:
            self.has = []

        if self.cfg.architecture == "unet":
            if self.cfg.goal != "gt_flow_pred":
                self.model = Unet(
                    64,  # number of dimensions within the Unet itself
                    channels=3 + 3,  # input dimension
                    out_dim=self.radius ** 2 + 1 * int("colweights" in self.has) + 3 * int("cols" in self.has),  # output
                    time_in=False
                )
            else:
                self.model = Unet(
                    64,  # number of dimensions within the Unet itself
                    channels=3 + 3,  # input dimension
                    out_dim=2,      # output
                    time_in=False
                )
        elif self.cfg.architecture == 'raft':
            self.model = RAFT(cfg)

        self.mask = torch.nn.functional.unfold(
            torch.ones((1, self.image_h, self.image_w)),
            (self.radius, self.radius),
            padding=self.radius // 2
        ).reshape((1, self.radius ** 2, self.image_h, self.image_w))
        if "colweights" in self.has:
            self.mask = torch.cat((self.mask, torch.ones(1, 1, self.image_h, self.image_w)), dim=1)

    def configure_optimizers(self):
        self.optimizers = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        return self.optimizers

    def apply_filter(self, fil, img, mode='softmax', flow_in='second'):
        if fil.shape[1] > 2:
            col = None
            if fil.shape[1] > self.radius ** 2 + 1:
                col = fil[:, -3:, :, :]  # additional color
                fil = fil[:, :-3]
            elif fil.shape[1] > self.radius ** 2:
                if self.cfg.cols == "ones":
                    col = torch.ones_like(fil[:, -3:, :, :])

            # Normalize filter
            if mode == 'softmax':
                fil_ = fil - torch.max(fil, dim=1, keepdim=True)[0]  # numerical stability
                fil_ = torch.exp(fil_) + self.cfg.eps
                fil_ = fil_ * self.mask.to(self.device)  # mask out the edge vals
                # safe division
                fil = fil_ / torch.sum(fil_, dim=1, keepdim=True)
            elif mode == 'mode':
                fil = torch.exp(fil) * self.mask.to(self.device)  # mask out the edge vals
                fil = torch.eq(fil, torch.max(fil, dim=1, keepdim=True)[0]).float()  # 1 at modes
                fil = fil / torch.sum(fil, dim=1, keepdim=True)  # normalize multiple modes
            elif mode == 'weighted_sum':
                denom = torch.sum(fil[:, :self.radius ** 2] * self.mask.to(self.device)[:, :self.radius ** 2], dim=1,
                                  keepdim=True)
                denom = torch.where(denom > self.cfg.eps, denom, torch.full(denom.shape, float('nan')).to(self.device))
                fil = fil / denom
            elif mode == 'none':
                pass
            orig_fil = fil if col is None else torch.cat((fil, col), dim=1)

            # Reshape weights
            fil, col_weight = (fil[:, :-1], fil[:, -1]) if "colweights" in self.has else (fil, None)
            fil = fil.reshape((-1, self.radius, self.radius, self.image_h, self.image_w))

            # Reshape image to apply filter
            flat = img.reshape((-1, self.image_h, self.image_w))  # batch the colors together
            unfold = torch.nn.functional.unfold(flat, (self.radius, self.radius), padding=self.radius // 2)
            unfold = unfold.reshape((-1, 3, self.radius, self.radius, self.image_h, self.image_w))

            applied = unfold * torch.unsqueeze(fil, 1)
            applied = torch.sum(applied, (2, 3))

            if torch.any(torch.isnan(applied)):  # Fill holes with neighbors
                blur = torchvision.transforms.GaussianBlur(kernel_size=self.radius, sigma=self.radius // 2)
                bg = blur(img)
                applied[torch.isnan(applied)] = bg[torch.isnan(applied)]

            if "cols" in self.has:
                applied = applied + col_weight[:, None] * col
            return (
                applied,  # filtered image
                orig_fil  # filter
            )
        elif flow_in == 'second':
            # copied from online
            B, C, H, W = img.size()
            # mesh grid
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float().to(self.device)

            fil = fil.flip(1)
            vgrid = grid + fil
            # scale grid to [-1,1]
            vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
            vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

            #return vgrid[:, None, 1, :, :].tile((1, 3, 1, 1)), fil

            vgrid = vgrid.permute(0, 2, 3, 1)
            output = torch.nn.functional.grid_sample(img, vgrid, align_corners=True)
            mask = torch.ones(img.size()).to(self.device)
            mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=True)

            mask[mask < 0.999] = 0
            mask[mask > 0] = 1

            red = torch.Tensor([1.0, 0.0, 0.0])[None, :, None, None].to(self.device)

            return output * mask + red * (1 - mask), fil
        elif flow_in == 'first':
            bsz = img.shape[0]
            imsize = img.shape[-1]

            currs = []
            weights = []
            flr, frac = torch.floor(fil), torch.where(torch.frac(fil) < 0.0, 1 + torch.frac(fil), torch.frac(fil))
            for fx, wx in zip((flr[:, 0, :, :], flr[:, 0, :, :] + 1), (1 - frac[:, 0, :, :], frac[:, 0, :, :])):
                for fy, wy in zip((flr[:, 1, :, :], flr[:, 1, :, :] + 1), (1 - frac[:, 1, :, :], frac[:, 1, :, :])):
                    x_idx = torch.arange(imsize).reshape((1, -1, 1)).tile((bsz, 1, imsize)).to(self.device)
                    x_idx_ = x_idx + fx
                    y_idx = torch.arange(imsize).reshape((1, 1, -1)).tile((bsz, imsize, 1)).to(self.device)
                    y_idx_ = y_idx + fy
                    b_idx = torch.arange(bsz).reshape((-1, 1, 1)).tile((1, imsize, imsize)).to(self.device)

                    x_idx, x_idx_, y_idx, y_idx_, b_idx = x_idx.long(), x_idx_.long(), y_idx.long(), y_idx_.long(), b_idx.long()

                    # Give a unique third index
                    unique_idx = x_idx_.reshape((bsz, -1)) * (3 * imsize) + y_idx_.reshape((bsz, -1))
                    sort_idx, perm = torch.sort(unique_idx, dim=-1)
                    lim = 10    # we can handle up to this many copies
                    stagger = torch.full((lim, bsz, imsize ** 2), float('nan')).to(self.device)
                    for l in range(1, lim):
                        stagger[l, :, l:] = sort_idx[:, :-l]
                    stagger[1:, :, :] = stagger[1:, :, :] - sort_idx[None, :, :]
                    stagger = torch.where(torch.eq(stagger, torch.zeros_like(stagger)),
                                          torch.ones_like(stagger),
                                          torch.full(stagger.shape, float('nan')).to(self.device))

                    l_idx = torch.nansum(stagger, dim=0)
                    batch_indexer = torch.arange(bsz)[:, None].tile((1, imsize ** 2)).to(self.device)
                    l_idx[batch_indexer, perm] = l_idx.clone()
                    l_idx = l_idx.reshape(x_idx_.shape).long()

                    # Bound
                    x_idx_ = torch.clamp(x_idx_, 0, imsize - 1)
                    y_idx_ = torch.clamp(y_idx_, 0, imsize - 1)

                    curr_ = torch.Tensor([float('nan')]).reshape((1, 1, 1, 1, 1)).tile((bsz, 3, imsize, imsize, lim)).to(self.device)
                    w = torch.Tensor([float('nan')]).reshape((1, 1, 1, 1)).tile((bsz, imsize, imsize, lim)).to(self.device)

                    curr_[b_idx, :, x_idx_, y_idx_, l_idx] = img[b_idx, :, x_idx, y_idx] * (wx * wy)[b_idx, None, x_idx, y_idx]
                    w[b_idx, x_idx_, y_idx_, l_idx] = (wx * wy)[b_idx, x_idx, y_idx]

                    #curr_ = torch.nanmean(curr_, dim=-1)
                    currs.append(curr_)
                    weights.append(w)

            bg = torch.Tensor([1.0, 0.0, 0.0]).reshape((1, 3, 1, 1)).tile((bsz, 1, imsize, imsize)).to(self.device)
            currs = torch.cat(currs, dim=-1)
            weights = torch.cat(weights, dim=-1)
            currs = torch.nansum(currs, dim=-1) / torch.nansum(weights, dim=-1)[:, None]

            currs = torch.where(torch.isnan(currs), bg, currs)

            return (
                currs,
                fil
            )

            """
            def warp_flow(im, flo):
                # Sizhe: I think img is of shape H, W, C, and flow is of shape: H, W, 2

                h, w = flo.shape[:2]
                flow_new = flo.copy()
                flow_new[:, :, 0] += np.arange(w)
                flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

                res = cv2.remap(
                    im, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
                )
                return res

            outs = []
            fil = fil.flip(1)
            for im, flo in zip(img, fil):
                im = im.permute((1, 2, 0)).cpu().numpy()
                flo = flo.permute((1, 2, 0)).detach().cpu().numpy()
                im = warp_flow(im, flo)
                im = torch.Tensor(im).permute((2, 0, 1)).to(self.device)
                outs.append(im)

            return torch.stack(outs, dim=0), fil
            """



    def invert_filter(self, fil):
        # send [dx, dy, x - dx, y - dy] to [-dx, -dy, x, y]
        fil_size = (1, self.radius, self.radius, self.image_h, self.image_w)
        reshaped_mask = self.mask[:, :self.radius ** 2].reshape(fil_size)

        fil = fil.clone()
        colw, cols = None, None
        if fil.shape[1] > self.radius ** 2:
            colw = fil[:, self.radius ** 2, None]
        if fil.shape[1] > self.radius ** 2 + 1:
            cols = fil[:, self.radius ** 2 + 1:]
        fil = fil[:, :self.radius ** 2].reshape((-1, self.radius, self.radius, self.image_h, self.image_w))

        dx = (torch.arange(0, self.radius) - self.radius // 2)[None, :, None, None, None]
        x_ = torch.arange(0, self.image_h)[None, None, None, :, None]
        dx = torch.broadcast_to(dx, fil_size).clone()
        x_ = torch.broadcast_to(x_, fil_size).clone()
        x = dx + x_

        dy = (torch.arange(0, self.radius) - self.radius // 2)[None, None, :, None, None]
        y_ = torch.arange(0, self.image_w)[None, None, None, None, :]
        dy = torch.broadcast_to(dy, fil_size).clone()
        y_ = torch.broadcast_to(y_, fil_size).clone()
        y = dy + y_

        idxs = torch.stack((dx, x_, x, dy, y_, y, reshaped_mask), dim=0)
        idxs = idxs.reshape((7, -1)).long()
        idxs = idxs[:, idxs[6] == 1]  # Only keep ones inside the image
        dx, x_, x, dy, y_, y, _ = tuple(torch.chunk(idxs, 7, dim=0))

        fil[:, self.radius // 2 - dx, self.radius // 2 - dy, x, y] = fil[:, self.radius // 2 + dx, self.radius // 2 + dy, x_, y_]
        fil = fil.reshape((fil.shape[0], self.radius ** 2, self.image_h, self.image_w))
        # I think this is kind of atrocious but whatever, lol
        if colw is not None:
            colw = -1 * colw

        if cols is not None:
            return torch.cat((fil, colw, cols), dim=1)
        elif colw is not None:
            return torch.cat((fil, colw), dim=1)
        else:
            return fil

    def vector_from_filter(self, fil):
        if fil.shape[1] == 2:
            return fil

        indices = torch.arange(self.radius).to(self.device) - self.radius // 2  # index values
        fil = fil[:, :self.radius ** 2]                                         # remove colors if they exist
        fil = fil.reshape((fil.shape[0], self.radius, self.radius, self.image_h, self.image_w))
        first = indices[None, :, None, None, None] * fil
        second = indices[None, None, :, None, None] * fil
        first, second = torch.sum(first, (1, 2)), torch.sum(second, (1, 2))
        return torch.stack((first, second), dim=1)

    def filter_from_vector(self, vec):
        vec = torch.round(vec)
        rad = torch.Tensor([self.radius // 2]).to(self.device)
        vec = torch.minimum(torch.maximum(vec, -rad), rad) + self.radius // 2
        vec = vec.long()

        batch_idx = torch.arange(vec.shape[0]).to(self.device)[:, None, None].long()
        x_idx = torch.arange(self.image_h)[None, :, None].to(self.device).long()
        y_idx = torch.arange(self.image_w)[None, None, :].to(self.device).long()
        idx_size = (vec.shape[0], self.image_h, self.image_w)

        batch_idx = torch.flatten(torch.broadcast_to(batch_idx, idx_size).clone())
        x_idx = torch.flatten(torch.broadcast_to(x_idx, idx_size).clone())
        y_idx = torch.flatten(torch.broadcast_to(y_idx, idx_size).clone())

        fx_idx = torch.flatten(vec[:, 0])
        fy_idx = torch.flatten(vec[:, 1])

        fil = torch.zeros((vec.shape[0], self.radius, self.radius, self.image_h, self.image_w)).to(self.device)
        fil[batch_idx, fx_idx, fy_idx, x_idx, y_idx] = 1.0
        fil = fil.reshape((vec.shape[0], self.radius ** 2, self.image_h, self.image_w))
        fil = self.invert_filter(fil)

        return fil

    def filter_to_image(self, filters):
        filters = filters[:self.radius ** 2]
        filters = filters.reshape((self.radius, self.radius, -1)).permute(2, 0, 1)
        filters = filters.reshape((-1, 1, self.radius, self.radius)).tile(1, 3, 1, 1)  # batch and rgb
        filters[:, 2, self.radius // 2, :] = 0.33 * (1 + 2 * filters[:, 2, self.radius // 2, :])  # tint centers
        filters[:, 1, :, self.radius // 2] = 0.33 * (1 + 2 * filters[:, 1, :, self.radius // 2])
        filters = filters.repeat_interleave(3, dim=2).repeat_interleave(3, dim=3)  # scale up
        return torchvision.utils.make_grid(filters, nrow=round(math.sqrt(filters.shape[0])), pad_value=1.0)

    # Note that this derivative is not signed
    def _derivative(self, mtx, dim, deg=1, sep=1, sides='one'):
        # Sep=2 in UFlow
        if deg == 1:
            ret = []
            for d in dim:
                indexer = torch.arange(sep, mtx.shape[d]).to(self.device)
                diffs = mtx.index_select(d, indexer) - mtx.index_select(d, indexer - sep)

                if sides == 'both':
                    nan_tile = list(mtx.shape)
                    nan_tile[d] = 1
                    nans = torch.Tensor([float('nan')])
                    nans = nans.to(self.device).reshape(tuple([1] * len(mtx.shape))).tile(*nan_tile)

                    diffs = torch.stack((torch.cat((diffs, nans), dim=d),
                                         torch.cat((nans, diffs), dim=d)), dim=0)
                    diffs = torch.abs(diffs)
                    ret.append(torch.nanmean(diffs, dim=0))
                else:
                    diffs = torch.stack([torch.squeeze(t, dim=d) for t in torch.chunk(diffs, diffs.shape[d], dim=d)], dim=0)
                    ret.append(diffs)
            return torch.stack(ret, dim=0)

    # Not edge aware -- take a look at UFlow's implementation
    def smoothness_loss(self, fil, target):
        vecs = self.vector_from_filter(fil)
        #dxy = torch.sqrt(torch.sum(self._derivative(vecs, (2, 3)).square(), dim=0)) # magnitude of derivative
        #dimg = torch.sqrt(torch.sum(self._derivative(target, (2, 3)).square(), dim=0))
        #return torch.mean(torch.exp(-self.cfg.smoothness_lmbd * torch.sum(dimg, dim=1)) * torch.sum(dxy, dim=1))

        dxy = torch.sum(torch.abs(self._derivative(vecs, (2, 3))), dim=3)   # sum across channels
        dimg = torch.sum(torch.abs(self._derivative(target, (2, 3))), dim=3)
        return torch.mean(torch.exp(-self.cfg.smoothness_lmbd * dimg) * dxy)

        #return torchmetrics.functional.total_variation(vecs, reduction='mean') / (2 * self.image_size ** 2)
        #return torch.mean(torch.sqrt(dxy))  # L1-loss

    def copout_loss(self, fil):
        if fil.shape[1] > self.radius ** 2:
            loss = torch.nn.functional.mse_loss(fil[:, self.radius ** 2], torch.zeros_like(fil[:, self.radius ** 2]))
            return loss
        else:
            return 0.0

    def corrective_loss(self, inp, target):
        inp = inp.reshape((inp.shape[0], -1))
        which_white = torch.eq(torch.min(inp, 1)[0], 1.0)

        white_ones = target[which_white, 0]
        missed_pixels = torch.sum(torch.eq(white_ones, 0.0).float())

        ret = -1 * missed_pixels / (self.image_w * self.image_h * inp.shape[0])
        return ret

    def identity_loss(self, fil):
        weights = torch.square(torch.arange(self.radius) - self.radius // 2)
        weights = weights[None, :] + weights[:, None]
        weights = weights.reshape((-1))
        weights = weights[None, :, None, None]

        loss = torch.mean(fil[:, :self.radius ** 2] * weights.to(self.device))
        return loss

    def divergence_loss(self, fil):
        fil = self.invert_filter(fil)
        if fil.shape[1] > self.radius ** 2:
            inflow = fil[:, self.radius ** 2]
        fil = fil[:, :self.radius ** 2, self.radius // 2 : -self.radius // 2, self.radius // 2: -self.radius // 2]

        div = torch.sum(fil, dim=1)

        """ We do need to find some way to compensate for the inflow, but this is a little weird.
        # Add inflow back into the values to minimize loss
        div = div.reshape((fil.shape[0], -1)).sort()[0]
        while torch.sum(inflow) > self.cfg.eps:
            number_min = torch.sum(torch.eq(div[:, 0, None], div), dim=0)               # number of minima
            step_min = div[torch.arange(div.shape[0]).long(), number_min] - div[:, 0]   # amount it needs to be increased
            contribution = torch.minimum(inflow, number_min * step_min) / number_min    # how much each minimum gets
            mask = torch.eq(div[:, 0, None], div)                                       # mask for minimum
            div = div + mask.float() * contribution[:, None]                            # increase minimum
            inflow = inflow - contribution * number_min                                 # decrease inflow
        """

        eps_grid = torch.full(div.shape, self.cfg.small_eps).to(self.device)           # bound the loss
        div = torch.maximum(div, eps_grid)
        div = torch.minimum(div, 1 / eps_grid)

        return torch.mean(div + torch.reciprocal(div)) - 2.0                           # encourage bijectiveness
        #return torch.mean(torch.nn.functional.relu(div - 1.0))                                      # penalize many to one

    def inversion_loss(self, fil, inp, target):
        inverted = self.invert_filter(fil).reshape((fil.shape[0], -1, self.image_h, self.image_w))
        out = self.apply_filter(inverted, target, mode='weighted_sum')[0]
        return torch.nn.functional.mse_loss(out, inp)

    def loss(self, out, fil, target, inp, flow):
        if self.cfg.goal == 'filter_pred':
            sub_losses = (
                    torch.nn.functional.mse_loss(out, target),                          # photometric loss
                    self.cfg.smoothness_weight * self.smoothness_loss(fil, target),   # smoothness loss
                    self.cfg.copout_weight * self.copout_loss(fil),                     # penalize for using color
                    self.cfg.identity_weight * self.identity_loss(fil),                 # encourage zeroes
                    self.cfg.divergence_weight * self.divergence_loss(fil),
                    self.cfg.inversion_weight * self.inversion_loss(fil, inp, target)
            )
            return (
                    sum(sub_losses),  # actual loss
                    sub_losses[0]                                                   # how off from ideal
            )
        elif self.cfg.goal == 'gt_filter_pred':
            #target_flow = self.filter_from_vector(flow)
            #loss = torch.nn.functional.mse_loss(fil, target_flow) * (self.radius ** 2)
            vec = self.vector_from_filter(self.invert_filter(fil))
            loss = torch.nn.functional.mse_loss(vec, flow)
            photometric = torch.nn.functional.mse_loss(out, target)
            return loss, photometric
            """
            mask = torch.sum(target, dim=1, keepdim=True) >= self.cfg.eps
            loss = torch.square(fil - target)
            return torch.nanmean(torch.where(
                mask.tile((1, fil.shape[1], 1, 1)),
                loss,
                torch.full(loss.shape, float('nan')).to(self.device)
            ))
            """
        elif self.cfg.goal == 'gt_flow_pred':
            loss = torch.nn.functional.mse_loss(fil, flow)
            photometric = torch.nn.functional.mse_loss(out, target)
            return loss, photometric

    def mode_to_flow(self, fil):
        fil = fil[:, :self.radius ** 2]
        idxs = torch.argmax(fil, dim=1)
        idxs = torch.stack((idxs // self.radius - self.radius // 2, idxs % self.radius - self.radius // 2), dim=1)
        return idxs

    def training_step(self, batch, batch_idx):
        o = self.model(2 * torch.cat((batch[0], batch[1]), dim=1) - 1.0)
        out = [o] if self.cfg.architecture == 'unet' else o
        errs = []
        for ot in out:
            ot, fil = self.apply_filter(ot, batch[0])
            err, photo_loss = self.loss(ot, fil, batch[1], batch[0], batch[2])
            errs.append(err)
        err = sum(errs) / len(errs)

        mean_flow = self.vector_from_filter(fil)
        dist_to_flow = torch.nn.functional.mse_loss(mean_flow, batch[2])
        if self.cfg.goal != 'gt_flow_pred':
            opt_fil = self.filter_from_vector(batch[2])
        else:
            opt_fil = batch[2]
        opt_result, _ = self.apply_filter(opt_fil, batch[0], mode='weighted_sum')
        opt_loss, opt_photo = self.loss(opt_result, opt_fil, batch[1], batch[0], batch[2])

        self.log_dict({
            "train/loss": err,
            "train/photo": photo_loss,
            "train/flow_err": dist_to_flow,
            "train/opt_loss": opt_loss,
            "train/opt_photo": opt_photo
        })
        #Self.log_grad_norm_stat()

        return err

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            out = self.model(2 * torch.cat((batch[0], batch[1]), dim=1) - 1.0)
            if self.cfg.architecture == 'raft':
                out = out[-1]

            out_sf, sfs = self.apply_filter(out, batch[0])
            err, photo_loss = self.loss(out_sf, sfs, batch[1], batch[0], batch[2])
            self.log_dict({
                "val/loss": err,
                "val/photometric": photo_loss
            })

            if self.cfg.goal != 'gt_flow_pred':
                out_md, modes = self.apply_filter(out, batch[0], mode='mode')
                err, photo_loss = self.loss(out_md, modes, batch[1], batch[0], batch[2])
                self.log_dict({
                    "val/mode_loss": err,
                    "val/mode_photometric": photo_loss
                })

            bsz = batch[0].shape[0]

            # prepare for wandb upload
            def chunk(x):
                x[:, 0, 0, 0] = x[:, 0, 0, 0] * 0.95                 # not completely white so wandb doesn't bug out
                return list(torch.chunk(x, bsz))

            # Flows
            mean_flow = self.vector_from_filter(sfs)
            self.log_dict({
                "val/flow_err": torch.nn.functional.mse_loss(mean_flow, batch[2])
            })
            mean_flow = torchvision.utils.flow_to_image(mean_flow) / 255.0
            gt_flow = torchvision.utils.flow_to_image(batch[2]) / 255.0
            if self.cfg.goal != 'gt_flow_pred':
                mode_flow = torchvision.utils.flow_to_image(self.mode_to_flow(modes).to(torch.float)) / 255.0
                gt2 = self.vector_from_filter(self.filter_from_vector(batch[2]))
                gt2_flow = torchvision.utils.flow_to_image(gt2) / 255.0

                invert = self.apply_filter(self.invert_filter(sfs), batch[1], mode='none')[0]

            opt_vecs = self.filter_from_vector(batch[2]) if self.cfg.goal != "gt_flow_pred" else batch[2]
            opt_result, _ = self.apply_filter(opt_vecs, batch[0], mode='weighted_sum')

            # Filter image
            max_size = 20
            scale = max(1, self.radius // max_size)

            self.logger.log_image(key='original', images=chunk(batch[0]), step=self.global_step)
            self.logger.log_image(key='target', images=chunk(batch[1]), step=self.global_step)

            self.logger.log_image(key='softmax_p', images=chunk(out_sf), step=self.global_step)
            self.logger.log_image(key='opt_p', images=chunk(opt_result), step=self.global_step)
            if self.cfg.goal != 'gt_flow_pred':
                self.logger.log_image(key='mode_p', images=chunk(out_md), step=self.global_step)
                self.logger.log_image(key='invert_p', images=chunk(invert), step=self.global_step)

            self.logger.log_image(key='mean_flow', images=chunk(mean_flow), step=self.global_step)
            self.logger.log_image(key='gt_flow', images=chunk(gt_flow), step=self.global_step)
            if self.cfg.goal != 'gt_flow_pred':
                self.logger.log_image(key='mode_flow', images=chunk(mode_flow), step=self.global_step)
                self.logger.log_image(key='gt2_flow', images=chunk(gt2_flow), step=self.global_step)

            """
            vecs = self.vector_from_filter(sfs)
            dxy = torch.sum(torch.abs(self._derivative(vecs, (2, 3))), dim=3)  # sum across channels
            dimg = torch.sum(torch.abs(self._derivative(batch[1], (2, 3))), dim=3)
            dxy_ = torch.zeros((bsz, 3, self.image_h, self.image_w)).to(self.device)
            dxy_[:, 0, :-1, :] = dxy[0].permute((1, 0, 2)) # 31 n 32 -> n 31 32
            dxy_[:, 1, :, :-1] = dxy[1].permute((1, 2, 0)) # 31 n 32 -> n 32 31
            dimg_ = torch.zeros((bsz, 3, self.image_h, self.image_w)).to(self.device)
            dimg_[:, 0, :-1, :] = dimg[0].permute((1, 0, 2))  # 31 n 32 -> n 31 32
            dimg_[:, 1, :, :-1] = dimg[1].permute((1, 2, 0))  # 31 n 32 -> n 32 31
            self.logger.log_image(key='dxy', images=chunk(dxy_), step=self.global_step)
            self.logger.log_image(key='dimg', images=chunk(dimg_), step=self.global_step)
            """

            """
            fils = sfs[0, :, ::scale, ::scale]
            fil_image = self.filter_to_image(fils)
            log_param = 0.01
            fil_log_image = self.filter_to_image(
                (torch.log(log_param + fils) - math.log(log_param)) / (math.log(log_param + 1.0) - math.log(log_param))
            )  # Log view (smoother)
            invert_fil = self.invert_filter(sfs)[0, :, ::scale, ::scale]
            invert_fil_image = self.filter_to_image(invert_fil)

            self.logger.log_image(key='filters_0', images=[fil_image, fil_log_image, invert_fil_image], step=self.global_step)
            """
            if "colweights" in self.has:
                self.logger.log_image(key='col_weight', images=chunk(sfs[:, self.radius ** 2, None]), step=self.global_step)
            if "cols" in self.has:
                self.logger.log_image(key='color', images=chunk(sfs[:, -3:]), step=self.global_step)

            # Video logging with background
            # This might not work with multiple GPUs -- check
            # Color border
            """
            video = torch.eye(3)[None, :, :, None, None].tile((bsz, 1, 1, self.image_size + 2, self.image_size + 2))
            video[:, :, :, 1:-1, 1:-1] = torch.stack((batch[0], batch[1], out_sf), dim=1)

            video = video.repeat_interleave(20, dim=1)                      # 20x slower
            video[:, :, :, 0, 0] = (torch.arange(60) / 60)[None, :, None]   # different frames, or else not slower
            video = torch.cat(torch.tensor_split(video, bsz, dim=0), 3)

            video = torch.minimum(video, torch.Tensor([1.0]))                               # cap it at one
            video = video.detach().cpu().numpy() * 255.0                    # correct type
            video = video.astype(dtype=np.uint8)
            wandb.log({
                "compare": wandb.Video(video),
                "trainer/global_step": self.global_step
            })
            """

            self.log_video(batch[0], batch[1], out_sf, key="compare")

            if self.cfg.goal == 'gt_flow_pred':
                self.log_video(gt_flow, mean_flow, key="flow_compare")
                """
                video = torch.stack((gt_flow, mean_flow), dim=1)
                video = video.repeat_interleave(20, dim=1)  # 20x slower
                video[:, :, :, 0, 0] = (torch.arange(40) / 40)[None, :, None]  # different frames, or else not slower
                video = torch.cat(torch.tensor_split(video, bsz, dim=0), 3)

                video = torch.minimum(video, torch.Tensor([1.0]).to(self.device))  # cap it at one
                video = video.detach().cpu().numpy() * 255.0  # correct type
                video = video.astype(dtype=np.uint8)
                wandb.log({
                    "flow_compare": wandb.Video(video),
                    "trainer/global_step": self.global_step
                })
                """

    def log_video(self, *imgs, key='video'):
        bsz = imgs[0].shape[0]
        H, W = imgs[0].shape[-2:]
        n = len(imgs)

        bg = torch.eye(3).repeat((n // 3 + 1, 1))[:n]
        video = bg[None, :, :, None, None].tile((bsz, 1, 1, H + 2, W + 2))
        video[:, :, :, 1:-1, 1:-1] = torch.stack(imgs, dim=1)

        video = video.repeat_interleave(20, dim=1)                      # 20x slower
        video[:, :, :, 0, 0] = (torch.arange(20 * n) / (20 * n))[None, :, None]   # different frames, or else not slower
        video = torch.cat(torch.tensor_split(video, bsz, dim=0), 4)

        video = torch.minimum(video, torch.Tensor([1.0]))                               # cap it at one
        video = video.detach().cpu().numpy() * 255.0                    # correct type
        video = video.astype(dtype=np.uint8)
        wandb.log({
            key: wandb.Video(video),
            "trainer/global_step": self.global_step
        })

    def log_grad_norm_stat(self):
        with torch.no_grad():
            grad_norms = []
            gpr = []  # gradient-to-parameter ratio
            print(self.named_parameters(), len(list(self.named_parameters())))
            print([param.grad for _, param in self.named_parameters()])
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
