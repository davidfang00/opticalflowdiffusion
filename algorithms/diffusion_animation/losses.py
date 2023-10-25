import torch

def photometric_loss(ref, past_warped, future_warped, occ):
    future_loss = torch.sum(occ[:, 0, None] * charbonnier(ref - future_warped))
    past_loss = torch.sum(occ[:, 1, None]* charbonnier(ref - past_warped))
    return (future_loss + past_loss)

def constant_velocity_loss(p_flow, f_flow):
    return torch.mean(charbonnier(p_flow + f_flow))

def edgeaware_smoothness1(image, flow, edge_weight = 20):
    image_grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
    image_grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]

    flow_grad_y = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    flow_grad_x = flow[:, :, :, 1:] - flow[:, :, :, :-1]

    y_weights = torch.exp(-edge_weight * torch.mean(image_grad_y**2, dim=1, keepdim=True)) 
    x_weights = torch.exp(-edge_weight * torch.mean(image_grad_x**2, dim=1, keepdim=True)) 
    
    flow_y = charbonnier(flow_grad_y)
    flow_x = charbonnier(flow_grad_x)

    loss = torch.sum(x_weights * flow_x) + torch.sum(y_weights * flow_y)
    return loss

def occlusion_smoothness(image, occ, edge_weight = 20):
    image_grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
    image_grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]

    occ_grad_y = occ[:, :, 1:, :] - occ[:, :, :-1, :]
    occ_grad_x = occ[:, :, :, 1:] - occ[:, :, :, :-1]

    y_weights = torch.exp(-edge_weight * torch.mean(image_grad_y**2, dim=1, keepdim=True)) 
    x_weights = torch.exp(-edge_weight * torch.mean(image_grad_x**2, dim=1, keepdim=True)) 
    
    occ_y = occ_grad_y ** 2
    occ_x = occ_grad_x ** 2

    loss = torch.sum(x_weights * occ_x) + torch.sum(y_weights * occ_y)
    return loss

def occlusion_prior(occ):
    return -1 * torch.sum(occ[:, 0] * occ[:, 1])

def charbonnier(x, alpha=0.5, eps=1e-3):
    return torch.pow(torch.square(x) + eps**2, alpha)

def min_per_pixel_loss(ref, past_warped, future_warped):
    future_loss = charbonnier(ref - future_warped)
    past_loss = charbonnier(ref - past_warped)

    min_per_pixel = torch.minimum(future_loss, past_loss)
    return torch.mean(min_per_pixel)

def total_loss(ref, past_warped, future_warped, p_flow, f_flow, occ):
    photo_loss = photometric_loss(ref, past_warped, future_warped, occ)
    smooth_loss = edgeaware_smoothness1(ref, p_flow) + edgeaware_smoothness1(ref, f_flow)
    occ_smooth_loss = occlusion_smoothness(ref, occ)
    occ_prior_loss = 0.05 * occlusion_prior(occ)

    # print(photo_loss, smooth_loss, occ_smooth_loss, occ_prior_loss)
    # const_vel_loss = constant_velocity_loss(p_flow, f_flow)

    return photo_loss + smooth_loss + occ_smooth_loss + occ_prior_loss

