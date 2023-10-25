from omegaconf import DictConfig, OmegaConf
import os
import random
from tqdm import tqdm
from pathlib import Path
import time

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

from PIL import Image

class TaiChiDataset(torchvision.datasets.VisionDataset):
    mean = None
    std = None
    def __init__(self, cfg: DictConfig, split='training', device='cpu', mod="0,0"):
        if split == "validation":
            split = "test"

        self.cfg = cfg
        self.split = split
        self.image_size = cfg.image_size
        self.device = device

        dir = str(Path.home()) + '/datasets/taichi/taichi/' + split 
        self.first_frames = []
        self.second_frames = []

        random.seed(14)

        for it, vid in enumerate(os.scandir(dir)):
            if random.random() < cfg.scale_down:
                frames = [vid.path + '/' + x for x in sorted(os.listdir(vid.path))]
                self.first_frames += frames[:-cfg.frame_distance]
                self.second_frames += frames[cfg.frame_distance:]
    
        if mod.split(",")[1] != "0":
            rem, mod = tuple(mod.split(","))
            rem, mod = int(rem), int(mod)
            print(rem, mod)
            self.first_frames = self.first_frames[rem::mod]
            self.second_frames = self.second_frames[rem::mod]

        self.transform = transforms.Compose([  # Import and resize
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size))
        ])

        if cfg.calculate_flows:
            self.calculate_flows(cfg)
        self.flows = [x.replace(split, split + '-flows2') for x in self.first_frames]

    def __getitem__(self, index, inner = False):
        if self.split == 'test' and not inner:
            return torch.cat([
                self.__getitem__(index + i * self.cfg.frame_distance, inner=True) for i in range(self.cfg.val_length)
            ], dim=0) # get consecutive frames for val
        else:
            first_frame = self.first_frames[index]
            second_frame = self.second_frames[index]
            flow = self.flows[index]

            first_frame = Image.open(first_frame)
            second_frame = Image.open(second_frame)

            first_frame = self.transform(first_frame)
            second_frame = self.transform(second_frame)

            flow_transforms = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size))
            ]) # and maybe a normalize
            flow = flow_transforms(torch.load(flow))

            ret = torch.cat((second_frame, first_frame, flow), axis=0)
            return ret

    def __len__(self):
        return len(self.flows)

    def _raft_inference(self, first, second, model_transforms):
        first_b = torch.Tensor(len(first), 3, self.image_size, self.image_size)
        second_b = torch.Tensor(len(second), 3, self.image_size, self.image_size)

        for idx, (a, b) in enumerate(zip(first, second)):
            first_b[idx] = self.transform(Image.open(a))
            second_b[idx] = self.transform(Image.open(b))

        first_b = first_b.to(self.device)
        second_b = second_b.to(self.device)

        first_b, second_b = model_transforms(first_b, second_b)

        return self.flow_model(first_b, second_b)[-1]

    def calculate_flows(self, cfg):
        if cfg.flow_method not in ['raft', 'pipsonly']:
            raise NotImplementedError('Only raft optical flow is supported for now')

        if cfg.flow_method == 'raft':
            self.flow_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
            self.flow_model.to(self.device)
            model_transforms = Raft_Large_Weights.DEFAULT.transforms()

        start = time.time()
        with torch.no_grad():
            random.shuffle(self.first_frames)
            for i in range(0, len(self.first_frames), cfg.flow_batch_size):
                print(f'Calculating flows... {i}/{len(self.first_frames)} -- {time.time() - start}', end='\r')
                batch_end = min(i + cfg.flow_batch_size, len(self.first_frames))
                first = self.first_frames[i:batch_end]
                second = self.second_frames[i:batch_end]

                if cfg.flow_method == 'raft':
                    flows = self._raft_inference(first, second, model_transforms)

                for j in range(batch_end - i):
                    path_to_add = self.first_frames[i + j].replace(self.split, self.split + '-flows2')
                    parent = path_to_add[:path_to_add.rfind('/')]
                    if not os.path.exists(parent):
                        os.makedirs(parent)
                    torch.save(flows[j].to('cpu'), path_to_add)
