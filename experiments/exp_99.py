import os
from typing import Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision

import pytorch_lightning as pl


from omegaconf import DictConfig
from algorithms.diffusion_animation import MatrixFlow, FlowPred, FlowDiffuser, FlowLearner, PWCLearner
from datasets.animation import FlyingChairsDataset, ArtificialDataset, KittiSingleDataset, BuckBunnyVideoDataset, SintelDataset
from .exp_base import BaseLightningExperiment, BasePytorchExperiment

class MatrixFlowExperiment(BaseLightningExperiment):
    """
    A video prediction experiment
    """
    compatible_algorithms = dict(
        matrix_flow=MatrixFlow,
        flow_pred=FlowPred,
        flow_diffuser=FlowDiffuser,
        flow_learner=FlowLearner,
        pwc_learner=PWCLearner
    )

    compatible_datasets = dict(
        flying_chairs=FlyingChairsDataset,
        artificial=ArtificialDataset,
        kitti_single=KittiSingleDataset,
        buck_bunny_video=BuckBunnyVideoDataset,
        sintel=SintelDataset
    )

    def _build_dataset(self, split: str) -> Optional[torch.utils.data.Dataset]:
        if split in ['training', 'test', 'validation']:
            return self.compatible_datasets[self.cfg.dataset.name](self.cfg.dataset,
                                                                    split=split,
                                                                    device=BasePytorchExperiment._get_device('auto'))
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")

