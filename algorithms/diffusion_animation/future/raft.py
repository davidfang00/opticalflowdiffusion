import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .raft_update import BasicUpdateBlock, SmallUpdateBlock
from .raft_extractor import BasicEncoder, SmallEncoder
from .raft_corr import CorrBlock, AlternateCorrBlock
from .raft_utils import bilinear_sampler, coords_grid, upflow8
#from .filter import ConvToFilter, FilterToConv

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, cfg):
        super(RAFT, self).__init__()
        self.radius = cfg.radius

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.flow_dim = 289
        self.corr_levels = 4
        self.corr_radius = 4

        self.dropout = 0

        self.alternate_corr = False

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=self.dropout)
        self.update_block = BasicUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=hdim, flow_dim=self.flow_dim)

        # Filter blocks
        self.f2c = None #FilterToConv(self.radius, out_dim=self.flow_dim)
        self.c2f = None #ConvToFilter(self.radius, in_dim=self.flow_dim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, -1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, -1, 8 * H, 8 * W)

    def vector_from_filter(self, fil):
        image_size = fil.shape[-1]
        indices = torch.arange(self.radius).to(fil.device) - self.radius // 2  # index values
        fil = fil[:, :self.radius ** 2]                                         # remove colors if they exist
        fil = fil.reshape((fil.shape[0], self.radius, self.radius, image_size, image_size))
        first = indices[None, :, None, None, None] * fil
        second = indices[None, None, :, None, None] * fil
        first, second = torch.sum(first, (1, 2)), torch.sum(second, (1, 2))
        return torch.stack((first, second), dim=1)

    def forward(self, images, iters=1, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        image1, image2 = images[:, :3], images[:, :3]

        #image1 = 2 * (image1 / 255.0) - 1.0
        #image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        flow = torch.full((image1.shape[0], self.radius ** 2, image1.shape[2] // 8, image1.shape[3] // 8), 0.5).to(image1.device)
        coords0, coords1 = self.initialize_flow(image1)
        assert((self.vector_from_filter(2 * flow - 1) == (coords1 - coords0)).all())

        flow_predictions = []
        for itr in range(iters):
            if itr > 0:
                coords1 = coords0 + self.vector_from_filter(2 * flow - 1)
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            net, delta_flow = self.update_block(net, inp, corr, self.f2c(flow))

            # F(t+1) = F(t) + \Delta(t)
            flow = flow + self.c2f(delta_flow)

            # upsample predictions
            #if up_mask is None:
            flow_up = upflow8(flow)
            #else:
            #    flow_up = self.upsample_flow(flow, up_mask)

            flow_predictions.append(2 * flow_up - 1)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
