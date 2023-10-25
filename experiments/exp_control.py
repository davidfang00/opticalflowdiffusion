import os
from typing import Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


from omegaconf import DictConfig
from datasets.animation import TaiChiDataset
from algorithms.diffusion_animation import FrameGenerator, FlowCompleter
from .exp_base import BaseLightningExperiment, BasePytorchExperiment


class AnimationExperiment(BaseLightningExperiment):
    """
    A video prediction experiment
    """
    compatible_algorithms = dict(
        frame_generator=FrameGenerator,
        flow_completer=FlowCompleter
    )

    compatible_datasets = dict(
        taichi=TaiChiDataset
    )

    def _build_dataset(self, split: str) -> Optional[torch.utils.data.Dataset]:
        if split in ['training', 'test', 'validation']:
            return self.compatible_datasets[self.cfg.dataset.name](self.cfg.dataset, split=split, device=BasePytorchExperiment._get_device('auto'))
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")

