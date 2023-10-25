from flying_chairs import FlyingChairsDataset

import torch, torchvision

import omegaconf

class MatrixFlow:
    def __init__(self, cfg):
        self.automatic_optimization = False
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.radius = cfg.radius
        assert (self.radius % 2 == 1)

        if "cols" in dir(cfg):
            if cfg.cols == "any":
                self.has = ["cols", "colweights"]
            else:
                self.has = ["colweights"]
        else:
            self.has = []

        self.mask = torch.nn.functional.unfold(
            torch.ones((1, self.image_size, self.image_size)),
            (self.radius, self.radius),
            padding=self.radius // 2
        ).reshape((1, self.radius ** 2, self.image_size, self.image_size))
        self.device = torch.device("cpu")
        if "colweights" in self.has:
            self.mask = torch.cat((self.mask, torch.ones(1, 1, self.image_size, self.image_size)), dim=1)

    def invert_filter(self, fil):
        # send [dx, dy, x - dx, y - dy] to [-dx, -dy, x, y]
        fil_size = (1, self.radius, self.radius, self.image_size, self.image_size)
        reshaped_mask = self.mask[:, :self.radius ** 2].reshape(fil_size)

        fil = fil.clone()
        colw, cols = None, None
        if fil.shape[1] > self.radius ** 2:
            colw = fil[:, self.radius ** 2, None]
        if fil.shape[1] > self.radius ** 2 + 1:
            cols = fil[:, self.radius ** 2 + 1:]
        fil = fil[:, :self.radius ** 2].reshape((-1, self.radius, self.radius, self.image_size, self.image_size))

        dx = (torch.arange(0, self.radius) - self.radius // 2)[None, :, None, None, None]
        x_ = torch.arange(0, self.image_size)[None, None, None, :, None]
        dx = torch.broadcast_to(dx, fil_size).clone()
        x_ = torch.broadcast_to(x_, fil_size).clone()
        x = dx + x_

        dy = (torch.arange(0, self.radius) - self.radius // 2)[None, None, :, None, None]
        y_ = torch.arange(0, self.image_size)[None, None, None, None, :]
        dy = torch.broadcast_to(dy, fil_size).clone()
        y_ = torch.broadcast_to(y_, fil_size).clone()
        y = dy + y_

        idxs = torch.stack((dx, x_, x, dy, y_, y, reshaped_mask), dim=0)
        idxs = idxs.reshape((7, -1)).long()
        idxs = idxs[:, idxs[6] == 1]  # Only keep ones inside the image
        dx, x_, x, dy, y_, y, _ = tuple(torch.chunk(idxs, 7, dim=0))

        fil[:, self.radius // 2 - dx, self.radius // 2 - dy, x, y] = fil[:, self.radius // 2 + dx, self.radius // 2 + dy, x_, y_]
        fil = fil.reshape((fil.shape[0], self.radius ** 2, self.image_size, self.image_size))
        # I think this is kind of atrocious but whatever, lol
        if colw is not None:
            colw = -1 * colw

        if cols is not None:
            return torch.cat((fil, colw, cols), dim=1)
        elif colw is not None:
            return torch.cat((fil, colw), dim=1)
        else:
            return fil

    def filter_from_vector(self, vec):
        vec = torch.round(vec)
        rad = torch.Tensor([self.radius // 2]).to(self.device)
        vec = torch.minimum(torch.maximum(vec, -rad), rad) + self.radius // 2
        vec = vec.long()

        batch_idx = torch.arange(vec.shape[0]).to(self.device)[:, None, None].long()
        idx = torch.arange(self.image_size).to(self.device).long()
        x_idx, y_idx = idx[None, :, None], idx[None, None, :]
        idx_size = (vec.shape[0], self.image_size, self.image_size)

        batch_idx = torch.flatten(torch.broadcast_to(batch_idx, idx_size).clone())
        x_idx = torch.flatten(torch.broadcast_to(x_idx, idx_size).clone())
        y_idx = torch.flatten(torch.broadcast_to(y_idx, idx_size).clone())

        fx_idx = torch.flatten(vec[:, 0])
        fy_idx = torch.flatten(vec[:, 1])

        fil = torch.zeros((vec.shape[0], self.radius, self.radius, self.image_size, self.image_size)).to(self.device)
        fil[batch_idx, fx_idx, fy_idx, x_idx, y_idx] = 1.0
        fil = fil.reshape((vec.shape[0], self.radius ** 2, self.image_size, self.image_size))
        fil = self.invert_filter(fil)

        return fil

    def apply_filter(self, fil, img, mode='weighted_sum'):
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
            denom = torch.where(denom > self.cfg.eps, denom, torch.full(denom.shape, float('nan')))
            fil = fil / denom
        elif mode == 'none':
            pass
        orig_fil = fil if col is None else torch.cat((fil, col), dim=1)

        # Reshape weights
        fil, col_weight = (fil[:, :-1], fil[:, -1]) if "colweights" in self.has else (fil, None)
        fil = fil.reshape((-1, self.radius, self.radius, self.image_size, self.image_size))

        # Reshape image to apply filter
        flat = img.reshape((-1, self.image_size, self.image_size))  # batch the colors together
        unfold = torch.nn.functional.unfold(flat, (self.radius, self.radius), padding=self.radius // 2)
        unfold = unfold.reshape((-1, 3, self.radius, self.radius, self.image_size, self.image_size))

        applied = unfold * torch.unsqueeze(fil, 1)
        applied = torch.sum(applied, (2, 3))

        if torch.any(torch.isnan(applied)): # Fill holes with neighbors
            blur = torchvision.transforms.GaussianBlur(kernel_size=self.radius, sigma=self.radius // 2)
            bg = blur(img)
            applied[torch.isnan(applied)] = bg[torch.isnan(applied)]

        if "cols" in self.has:
            applied = applied + col_weight[:, None] * col
        return (
            applied,  # filtered image
            orig_fil  # filter
        )


# Load dataset
cfg = omegaconf.OmegaConf.load('../../configurations/dataset/flying_chairs.yaml')
dataset = FlyingChairsDataset(cfg, split='validation')
dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

cfg = omegaconf.OmegaConf.load('../../configurations/algorithm/matrix_flow.yaml')
mf = MatrixFlow(cfg)

for batch in dl:
    prev, curr, flow = batch
    imsize = prev.shape[-1]
    bsz = prev.shape[0]

    # Warp prev according to flow
    """
    curr_ = torch.Tensor([float('nan')]).reshape((1, 1, 1, 1, 1)).tile((bsz, 3, imsize, imsize, imsize ** 2))
    bg = torch.Tensor([1.0, 0.0, 0.0]).reshape((1, 3, 1, 1)).tile((bsz, 1, imsize, imsize))

    x_idx = torch.arange(imsize).reshape((1, -1, 1)).tile((bsz, 1, imsize))
    x_idx_ = x_idx + torch.round(flow[:, 0, :, :])
    y_idx = torch.arange(imsize).reshape((1, 1, -1)).tile((bsz, imsize, 1))
    y_idx_ = y_idx + torch.round(flow[:, 1, :, :])
    b_idx = torch.arange(bsz).reshape((-1, 1, 1)).tile((1, imsize, imsize))
    l_idx = torch.arange(imsize ** 2).reshape((1, imsize, imsize)).tile((bsz, 1, 1))

    x_idx, x_idx_, y_idx, y_idx_, b_idx, l_idx = x_idx.long(), x_idx_.long(), y_idx.long(), y_idx_.long(), b_idx.long(), l_idx.long()
    x_idx, x_idx_, y_idx, y_idx_, b_idx, l_idx = torch.chunk(torch.stack((x_idx, x_idx_, y_idx, y_idx_, b_idx, l_idx), dim=0).reshape((6, -1)), 6, dim=0)

    x_idx_ = torch.where(x_idx_ < 0, torch.zeros_like(x_idx_), x_idx_)
    x_idx_ = torch.where(x_idx_ >= imsize, (imsize - 1) * torch.ones_like(x_idx_), x_idx_)
    y_idx_ = torch.where(y_idx_ < 0, torch.zeros_like(y_idx_), y_idx_)
    y_idx_ = torch.where(y_idx_ >= imsize, (imsize - 1) * torch.ones_like(y_idx_), y_idx_)

    curr_[b_idx, :, x_idx_, y_idx_, l_idx] = prev[b_idx, :, x_idx, y_idx]
    curr_ = torch.nanmean(curr_, dim=-1)
    curr_ = torch.where(torch.isnan(curr_), bg, curr_)
    """

    """
    grid = flow.permute((0, 2, 3, 1)).flip(3) + torch.stack((
        torch.arange(imsize).reshape((1, 1, -1)).tile((bsz, imsize, 1)),
        torch.arange(imsize).reshape((1, -1, 1)).tile((bsz, 1, imsize)),
    ), dim=-1)
    grid = grid / torch.Tensor([imsize, imsize])[None, None, None, :] * 2 - 1

    curr_ = torch.nn.functional.grid_sample(prev, grid)
    """

    fil = mf.filter_from_vector(flow)
    curr_ = mf.apply_filter(fil, prev)[0]

    flow_img = torchvision.utils.flow_to_image(flow).float() / 255.0
    flow_frac_img = torchvision.utils.flow_to_image(torch.frac(flow)).float() / 255.0
    flow_x = flow[:, 0, None, :, :].tile((1, 3, 1, 1))
    flow_x = (flow_x - torch.min(flow_x)) / (torch.max(flow_x) - torch.min(flow_x))
    flow_y = flow[:, 1, None, :, :].tile((1, 3, 1, 1))
    flow_y = (flow_y - torch.min(flow_y)) / (torch.max(flow_y) - torch.min(flow_y))
    print(torch.max(flow_img), torch.min(flow_img))
    print(torch.max(prev), torch.min(prev))
    img = torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(
        torch.cat((flow_img, flow_frac_img, flow_x, flow_y, torch.arange(imsize).reshape((1, 1, -1, 1)).tile((bsz, 3, 1, imsize)), 0.5 * flow_img + 0.5 * prev, prev, curr, curr_), dim=0)
    ))
    img.save('result.png')
    input("Ready...")