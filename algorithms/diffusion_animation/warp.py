from .softsplat_new import softsplat

import math

import torchvision
import torch

def get_radius(flow, C=3):
    R = math.sqrt(flow.shape[1] - C - 1)
    assert(int(R) - R < 1e-6 and int(R) % 2 == 1)
    R = int(R)

    return R

def unpack_flow(flow, C=3):
    H, W = flow.shape[-2], flow.shape[-1]
    R = get_radius(flow, C=C)
    return (
        flow[:, :-1-C].reshape((-1, R, R, H, W)),
        flow[:, -1-C:-1],
        flow[:, -1, None]
    )

def pack_flow(*flow):
    flow = list(flow)
    flow[0] = flow[0].reshape((flow[0].shape[0], -1, flow[0].shape[-2], flow[0].shape[-1]))
    return torch.cat(flow, dim=1)

def bound_mask(flow):
    R = get_radius(flow)
    H, W = flow.shape[-2], flow.shape[-1]

    mask = torch.nn.functional.unfold(
            torch.ones((1, H, W)),
            (R, R),
            padding=R // 2
    ).reshape((1, R, R, H, W)).to(flow.device)

    return mask

def invert_filter(flow):
    flow = flow.clone()

    R = get_radius(flow)
    fil, col, colw = unpack_flow(flow)
    H, W = flow.shape[-2], flow.shape[-1]
    fil_size = (1, R, R, H, W)
    mask = bound_mask(flow).to('cpu')
    
    dx = (torch.arange(0, R) - R // 2)[None, :, None, None, None]
    x_ = torch.arange(0, H)[None, None, None, :, None]
    dx = torch.broadcast_to(dx, fil_size).clone()
    x_ = torch.broadcast_to(x_, fil_size).clone()
    x = dx + x_

    dy = (torch.arange(0, R) - R // 2)[None, None, :, None, None]
    y_ = torch.arange(0, W)[None, None, None, None, :]
    dy = torch.broadcast_to(dy, fil_size).clone()
    y_ = torch.broadcast_to(y_, fil_size).clone()
    y = dy + y_

    idxs = torch.stack((dx, x_, x, dy, y_, y, mask), dim=0)
    idxs = idxs.reshape((7, -1)).long().to(flow.device)
    idxs = idxs[:, idxs[6] == 1]
    dx, x_, x, dy, y_, y, _ = tuple(torch.chunk(idxs, 7, dim=0))

    fil[:, R // 2 - dx, R // 2 - dy, x, y] = fil[:, R // 2 + dx, R // 2 + dy, x_, y_]
    return pack_flow(fil, col, colw)

def filter_to_flow(flow):
    R = get_radius(flow)
    H, W = flow.shape[-2], flow.shape[-1]
    fil, col, colw = unpack_flow(flow)

    indices = torch.arange(R) - R // 2
    indices = indices.to(flow.device)

    y = indices[None, :, None, None, None] * fil
    x = indices[None, None, :, None, None] * fil
    x, y = torch.sum(x, (1, 2)), torch.sum(y, (1, 2))
    return torch.stack((x, y), dim=1)

def warp(first, second, flow, rep='flow', mode='backward', **kwargs):
    if rep == 'flow':
        if mode == 'backward':
            return warp_backward_flow(first, second, flow, **kwargs)
        elif mode == 'forward':
            return warp_forward_flow(first, second, flow, **kwargs)
    elif rep == 'filter':
        if mode == 'backward':
            return warp_backward_filter(first, second, flow, **kwargs)
        elif mode == 'forward':
            return warp_forward_filter(first, second, flow, **kwargs)

def warp_backward_flow(first, second, flow):
    # copied from online
    B, C, H, W = second.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(second.device)

    flow = flow.flip(1)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(second, vgrid, align_corners=True)
    mask = torch.ones(second.size()).to(second.device)
    mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output, mask  # * mask + red * (1 - mask)

def warp_forward_flow(first, second, flow, scale=1, set_nans=True, get_variance=False, offset=[0, 0], warp_style="sum"):
    first = first.clone()
    weights = torch.ones_like(first[:, 0, :, :])
    where_nans = torch.isnan(first)
    first[where_nans] = 0.0
    weights[torch.any(where_nans, dim=1)] = 0.0
    if get_variance:
        var_weights = weights.clone()

    offset = [o % scale for o in offset]
    ret = softsplat(
        first,
        flow,
        weights[:, None],
        "linear_unn" if warp_style == "sum" else "linear",
        scale,
        offset
    )
    img = ret[:, :-1, :, :]
    weights = ret[:, -1, None, :, :].repeat((1, img.shape[1], 1, 1))

    if get_variance:
        var = softsplat(
            torch.square(first),
            flow,
            var_weights[:, None],
            "linear_unn",
            scale,
            offset
        )
        var_ = var[:, :-1, :, :]
        img = var_ - torch.square(img)

    if set_nans:
        img = torch.where(weights > 0, img, torch.full_like(img, float('nan')))
    return img

def warp_backward_filter(first, second, flow):
    B, C, H, W = second.size()
    R = get_radius(flow)
    mask = bound_mask(flow)

    fil, col, colw = unpack_flow(flow)
    fil = fil * mask

    flat = second.reshape((B * C, H, W))
    unfold = torch.nn.functional.unfold(flat, (R, R), padding = R // 2)
    unfold = unfold.reshape((B, C, R, R, H, W))

    applied = unfold * torch.unsqueeze(fil, 1)
    applied = torch.sum(applied, (2, 3))

    applied = applied + col * colw

    return applied

def warp_forward_filter(first, second, flow):
    flow = invert_filter(flow)
    return warp_backward_filter(second, first, flow)

def permute_warp(first, flow, scale_up=True):
    pass


    """
    # one to one warp
    # it's not one to one; there's a bug in the one to one we need to fix TO DO
    #print('permute warp! ')
    B, C, H, W = second.size()
    #flow[:, 0, :, :] = flow[:, 0, :, :] * H
    #flow[:, 1, :, :] = flow[:, 1, :, :] * W
    #cflow = torch.zeros_like(flow)
    flow = flow #/ 1e+3
    #print(flow.max(), flow.min())

    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(second.device)

    flow = flow.flip(1)
    vgrid = grid
    vgrid[:, 0, :, :] = vgrid[:, 0, :, :].clone() / max(W - 1, 1)
    vgrid[:, 1, :, :] = vgrid[:, 1, :, :].clone() / max(H - 1, 1)
    vgrid = vgrid + flow

    # wrap around
    vgrid = vgrid - torch.floor(vgrid)

    # sort rows
    vgrid = vgrid.reshape((B, 2, -1))
    row_idx = torch.floor(torch.argsort(vgrid[:, 0, :], dim=-1) / W)
    col_idx = torch.argsort(vgrid[:, 1, :] + 5 * row_idx, dim=-1) % W
    vgrid = torch.stack((row_idx, col_idx), dim=1).reshape((B, 2, H, W))


    # grid sample
    #vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    #vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = 2.0 * vgrid.clone() - 1.0
    vgrid = vgrid.permute((0, 2, 3, 1))


    ret = torch.nn.functional.grid_sample(second, vgrid, align_corners=True)
    #print(torch.isclose(ret, second).float().mean())
    return ret
    """





def scale(img, up=None, down=None):
    if up != None and down != None:
        raise ValueError('one of up or down')
    elif up != None:
        return torch.nn.functional.interpolate(img, scale_factor=up, mode='bilinear')
    elif down != None:
        patches = img.reshape((img.shape[0], img.shape[1], img.shape[2] // down, down, img.shape[3] // down, down))
        return torch.mean(torch.mean(patches, dim=-1), dim=-2)
    else:
        return img

def downsampled_warp(img, flow, warp_func, level=1):
    warped = []
    for i in range(0, level):
        for j in range(0, level):
            selection = img[:, :, i::level, j::level]

            selection_flow = flow[:, :, i::level, j::level]
            selection_flow = selection_flow / level
            
            warped.append(warp_func(selection, selection_flow))
    warped = torch.stack(warped, dim=0)
    warped = torch.mean(warped, 0)

    return warped

def nan_mse(pred, target, reduction='mean'):
    pred, target = pred.clone(), target.clone()
    pred = pred.flatten()
    target = target.flatten()
    where_not_nan = torch.logical_not(torch.logical_or(torch.isnan(target), torch.isnan(pred)))
    pred = pred[where_not_nan]
    target = target[where_not_nan]

    if reduction == 'mean':
        return torch.nanmean(torch.square(pred - target))
    elif reduction == 'none':
        return torch.square(pred - target)
    
def fill_holes_nan(img, weights):
    weights = weights.repeat((1, img.shape[1], 1, 1))
    filled_img = torch.where(weights > 0, img, torch.full_like(img, float('nan')))
    return filled_img

def charbonnier(x, alpha=0.5, eps=1e-3):
    return torch.pow(torch.square(x) + eps**2, alpha)

def nan_charbonnier(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    where_not_nan = torch.logical_not(torch.logical_or(torch.isnan(target), torch.isnan(pred)))
    pred = pred[where_not_nan]
    target = target[where_not_nan]
    return torch.mean(charbonnier(pred - target))

def edgeaware_smoothness1(image, flow, edge_weight = 30):
    image_grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
    image_grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]

    flow_grad_y = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    flow_grad_x = flow[:, :, :, 1:] - flow[:, :, :, :-1]

    y_weights = torch.exp(-edge_weight * torch.mean(image_grad_y**2, dim=1, keepdim=True)) 
    x_weights = torch.exp(-edge_weight * torch.mean(image_grad_x**2, dim=1, keepdim=True)) 
    
    flow_y = charbonnier(flow_grad_y)
    flow_x = charbonnier(flow_grad_x)

    loss = torch.mean(x_weights * flow_x) + torch.mean(y_weights * flow_y)
    return loss / 2

def spatial_smoothness_loss(flow):
    # Calculate the gradient along the height and width dimensions
    grad_height = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    grad_width = flow[:, :, :, 1:] - flow[:, :, :, :-1]

    # You can use either the L1 or L2 norm for the gradients.
    # L1 norm (absolute differences)
    loss_height = torch.abs(grad_height).mean()
    loss_width = torch.abs(grad_width).mean()

    # Alternatively, you could use the L2 norm (squared differences)
    # loss_height = (grad_height ** 2).mean()
    # loss_width = (grad_width ** 2).mean()

    # Combine the losses along both dimensions
    loss = loss_height + loss_width

    return loss
