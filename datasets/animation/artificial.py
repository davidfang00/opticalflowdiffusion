from omegaconf import DictConfig, OmegaConf

import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image


class ArtificialDataset(torchvision.datasets.VisionDataset):
    def __init__(self, cfg: DictConfig, split='training', device='cpu'):
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.size = cfg.size
        if 'seed' in dir(self.cfg):
            torch.manual_seed(cfg.seed) # Note that 'seed' option makes same train/test

        # Initial positions
        self.initial = torch.rand((self.cfg.size, 2, 1, 1))
        self.initial = (self.initial * self.image_size).long()

        if self.cfg.shape == 'boxes':
            self.wh = torch.rand((self.cfg.size, 2, 1, 1))
            self.wh = (self.wh * self.image_size).long()
        elif self.cfg.shape == 'squares':
            self.wh = torch.rand((self.cfg.size, 1, 1, 1))
            self.wh = (self.wh * self.image_size).long()
            self.wh = self.wh.tile((1, 2, 1, 1))
        elif self.cfg.shape == 'pixel':
            self.wh = torch.ones((self.cfg.size, 2, 1, 1)).long()
        elif self.cfg.shape == '2by1':
            self.wh = torch.ones((self.cfg.size, 2, 1, 1)).long()
            self.wh[:, 0, :, :] = 2

        # Movements
        self.flows = torch.rand((self.size, 2, 1, 1))
        self.flows = (self.flows * 3).long() - 1

    def __getitem__(self, index, inner=False):  # this might have the longest indices ever...
        bg = None
        if self.cfg.bg == 'white':
            bg = torch.ones((3, self.image_size, self.image_size))
        elif self.cfg.bg == 'checkers':
            bg = torch.ones((3, self.image_size, self.image_size))
            bg[:, ::2, ::2] = 0.2
            bg[:, ::4, ::4] = 0.4

        # Fixed shapes
        first = bg.clone().tile((1, 2, 2))
        first[:,
            self.initial[index, 0, :, :]:self.initial[index, 0, :, :] + self.wh[index, 0, :, :],
            self.initial[index, 1, :, :]:self.initial[index, 1, :, :] + self.wh[index, 1, :, :]] = 0

        second = torch.ones((3,
                             self.image_size * 2 + 2,
                             self.image_size * 2 + 2))  # pad it a bit
        second[:, 1:-1, 1:-1] = bg.clone().tile((1, 2, 2))
        second[:,
            self.initial[index, 0, :, :] + self.flows[index, 0, :, :] + 1:self.initial[index, 0, :, :] + self.flows[index, 0, :, :] + self.wh[index, 0, :, :] + 1,
            self.initial[index, 1, :, :] + self.flows[index, 1, :, :] + 1:self.initial[index, 1, :, :] + self.flows[index, 1, :, :] + self.wh[index, 1, :, :] + 1] = 0

        flows = torch.zeros((2, self.image_size * 2, self.image_size * 2), dtype=torch.float)
        for dim in [0, 1]:
            flows[dim,
                self.initial[index, 0, :, :]:self.initial[index, 0, :, :] + self.wh[index, 0, :, :],
                self.initial[index, 1, :, :]:self.initial[index, 1, :, :] + self.wh[index, 1, :, :]] = self.flows[index, dim].float()

        first = first[:, :self.image_size, :self.image_size]
        second = second[:, 1:-1 - self.image_size, 1:-1 - self.image_size]
        flows = flows[:, :self.image_size, :self.image_size]

        return first, second, flows                                         # return first, second, flow

    def __len__(self):
        return self.cfg.size
