from omegaconf import DictConfig, OmegaConf
import os
import random
from tqdm import tqdm
import time

import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image


class FlyingChairsDataset(torchvision.datasets.VisionDataset):
    def __init__(self, cfg: DictConfig, split='training', device='cpu'):
        if split == "training":
            split = "train"
        else:
            split = "val"
        self.cfg = cfg
        self.imsz = [int(x) for x in cfg.image_size.split(",")]
        self.split = split

        self.dataset = torchvision.datasets.FlyingChairs("/home/davidfang/Research/Sitzmann/datasets",
                                                         split=split,
                                                         transforms=FlyingChairsDataset.tuple_to_tensor)
        self.resize = transforms.Resize((self.imsz[0], self.imsz[1]), antialias=False)
        self.resize_ = transforms.Resize((self.imsz[0], self.imsz[1]), antialias=False, interpolation=Image.NEAREST)

    @staticmethod
    def tuple_to_tensor(*tup):
        t = torchvision.transforms.ToTensor()
        return t(tup[0]), t(tup[1]), torch.as_tensor(tup[2]), tup[3]
    
    def normalize(im):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        return normalize(im)

    def __getitem__(self, index, inner=False):
        size = list(reversed(self.dataset[index][0].shape))[:2]
        #print(size)
        ret = [self.resize(t) for t in self.dataset[index][:2]] + [self.resize_(self.dataset[index][-1])]
        ret[2] = ret[2].float() / torch.Tensor(size)[:, None, None]

        # ret[2] = ret[2].flip(0) # flip the flow; this makes it compatible with my operations
        ret[2] = torch.Tensor(self.imsz)[:, None, None] * ret[2]#.float() / torch.Tensor(size)[:, None, None]  # scale the flow
        return ret

    def __len__(self):
        return self.dataset.__len__()
