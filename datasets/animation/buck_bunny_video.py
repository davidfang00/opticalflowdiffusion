import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

class BuckBunnyVideoDataset(Dataset):
    def __init__(self, cfg: DictConfig, split='training', device = 'cpu'):

        self.cfg = cfg
        self.imsz = [int(x) for x in cfg.image_size.split(",")]
        self.video_file = "/home/davidfang/Research/Sitzmann/datasets/BigBuckBunnyVideo/big_buck_bunny_720p_5mb.mp4"

        # self.video_file = video_file
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.frameskip = 1

        self.vidcap = cv2.VideoCapture(self.video_file)
        self.count = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(self.count)

        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, 10)
        success, self.image1 = self.vidcap.read()
        print(f"got frame 1")
        
        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, 10 + self.frameskip)
        success, self.image2 = self.vidcap.read()
        print(f"got frame 2")

        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, 10 + 2 * self.frameskip)
        success, self.image3 = self.vidcap.read()
        print(f"got frame 3")

    def __len__(self):
        return self.count - (2 * self.frameskip)

    def __getitem__(self, idx):
        # print(f"{idx}: start get item")
        
        # self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, 10)
        # print(f"{idx}success")

        # success, image1 = self.vidcap.read()
        # print(f"{idx}: got frame 1")
        
        # self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, 10 + self.frameskip)
        # success, image2 = self.vidcap.read()
        # print(f"{idx}: got frame 2")

        # self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, 10 + 2 * self.frameskip)
        # success, image3 = self.vidcap.read()
        # print(f"{idx}: got frame 3")

        image1 = self.image1
        image2 = self.image2
        image3 = self.image3
        
        # Convert the images from BGR to RGB
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

        # Resize the images to a fixed size (e.g., 128x128)
        image1 = cv2.resize(image1, (self.imsz[0], self.imsz[1]))
        image2 = cv2.resize(image2, (self.imsz[0], self.imsz[1]))
        image3 = cv2.resize(image3, (self.imsz[0], self.imsz[1]))

        # Convert the images to PyTorch tensors and normalize
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        image3 = self.transform(image3)

        return image1, image2, image3
        
        # return torch.stack((image1, image2, image3), dim=0)