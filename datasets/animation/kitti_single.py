from omegaconf import DictConfig, OmegaConf
import os
import random
from tqdm import tqdm
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import cv2 as cv

from PIL import Image


class KittiSingleDataset(torchvision.datasets.VisionDataset):
    def __init__(self, cfg: DictConfig, split='training', device='cpu'):
        if split == "training":
            split = "train"
        else:
            split = "val"
            
        self.cfg = cfg
        self.imsz = [int(x) for x in cfg.image_size.split(",")]
        self.split = split

        self.dataset = torchvision.datasets.KittiFlow(f"../datasets/KITTI/{split}",
                                                         split="train",
                                                         transforms=KittiSingleDataset.tuple_to_tensor)
        self.resize = transforms.Resize((self.imsz[0], self.imsz[1]), antialias=False)
        self.resize_ = transforms.Resize((self.imsz[0], self.imsz[1]), antialias=False, interpolation=Image.NEAREST)

    @staticmethod
    def tuple_to_tensor(*tup):
        t = torchvision.transforms.ToTensor()

        ret = np.zeros(tup[2].shape)
        ret[0, :, :] = cv.inpaint(tup[2][0, :, :, None], np.logical_not(tup[3]).astype(np.uint8)[..., None], 20.0, 0)
        ret[1, :, :] = cv.inpaint(tup[2][1, :, :, None], np.logical_not(tup[3]).astype(np.uint8)[..., None], 20.0, 0)

        return t(tup[0]), t(tup[1]), torch.as_tensor(ret), torch.as_tensor(tup[3])
        #return t(tup[1]), t(tup[0]), torch.as_tensor(tup[2]), tup[3]

    def __getitem__(self, index, inner=False):
        size = list(reversed(self.dataset[index][0].shape))[:2]
        #print(size)
        #print(self.dataset[index], [x.shape for x in self.dataset[index]])
        ret = [self.resize(t) for t in self.dataset[index][:2]] + [self.resize_(self.dataset[index][2])]
        ret[2] = ret[2].float() / torch.Tensor(size)[:, None, None]

        ret[2] = ret[2].flip(0) # flip the flow; this makes it compatible with my operations
        ret[2] = torch.Tensor(self.imsz)[:, None, None] * ret[2]#.float() / torch.Tensor(size)[:, None, None]  # scale the flow
        return ret

    def __len__(self):
        return self.dataset.__len__()
