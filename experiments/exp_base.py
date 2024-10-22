from abc import ABC, abstractmethod
from typing import Optional, Union, Literal, List, Dict
import pathlib
import os

import wandb
import hydra
import torch
from pytorch_lightning.strategies.ddp import DDPStrategy
import numpy as np
import torch.nn as nn
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from omegaconf import DictConfig


class BaseExperiment(ABC):
    """
    Abstract class for an experiment. This generalizes the pytorch lightning Trainer & lightning Module to more
    flexible experiments that doesn't fit in the typical ml loop, e.g. multi-stage reinforcement learning benchmarks.
    """
    def __init__(
        self,
        cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None
    ) -> None:
        """
        Constructor

        Args:
            cfg: configuration file that contains everything about the experiment
            logger: a pytorch-lightning WandbLogger instance
            ckpt_path: an optional path to saved checkpoint
        """
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        self.ckpt_path = ckpt_path

    def exec_task(self, task: str) -> None:
        """
        Executing a certain task specified by string. Each task should be a stage of experiment.
        In most computer vision / nlp applications, tasks should be just train and test.
        In reinforcement learning, you might have more stages such as collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """
        if task == 'train':
            self.train()
        else:
            raise ValueError(f"Specified task '{task}' not implemented for class {self.__class__.__name__}.")

    @abstractmethod
    def train(self) -> None:
        """
        All train happens here
        """

        raise NotImplementedError


class BasePytorchExperiment(BaseExperiment, ABC):
    """
    Base class for a pytorch based experiment. This class is designed to allow maximum flexibility.
    If you application is as simple as a computer vision / nlp train, you should use BaseLightningExperiment
    """

    def __init__(
        self,
        cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None
    ) -> None:
        super().__init__(cfg, logger, ckpt_path)
        self.device = self._get_device('auto')

    @staticmethod
    def _get_device(device) -> torch.device:
        """
        Get a pytorch device object from string. Accepts auto in particular
        :param device:
        :return:
        """
        if device == 'auto':
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                # mac
                device = 'mps'
            elif torch.cuda.is_available() and torch.backends.cuda.is_built():
                # cuda
                device = 'cuda'
            else:
                device = 'cpu'
        print(f'Using device {device}.')

        return torch.device(device)


class BaseLightningExperiment(BaseExperiment):
    """
    Abstract class for pytorch lightning experiments. Useful for computer vision & nlp where main components are
    simply models, datasets and train loop.
    """
    compatible_algorithms: Dict = NotImplementedError
    compatible_datasets: Dict = NotImplementedError

    def __init__(
        self,
        cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None
    ) -> None:
        super().__init__(cfg, logger, ckpt_path)
        self.model = self._build_model()

    def _build_trainer_callbacks(self):
        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))

    def _build_model(self):
        """
        Build the lightning module
        :return:  a pytorch-lightning module to be launched
        """
        return self.compatible_algorithms[self.cfg.algorithm.name](self.cfg.algorithm)

    def _build_training_loader(self) -> Optional[Union[TRAIN_DATALOADERS, LightningDataModule]]:
        train_dataset = self._build_dataset("training")
        if train_dataset:
            return torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.experiment.training.data.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.experiment.training.data.num_workers),
                shuffle=self.cfg.experiment.training.data.shuffle,
            )
        else:
            return None

    def _build_validation_loader(self) -> Optional[Union[TRAIN_DATALOADERS, LightningDataModule]]:
        validation_dataset = self._build_dataset("validation")
        if validation_dataset:
            return torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=self.cfg.experiment.validation.data.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.experiment.validation.data.num_workers),
                shuffle=self.cfg.experiment.validation.data.shuffle,
            )
        else:
            return None

    def _build_test_loader(self) -> Optional[Union[TRAIN_DATALOADERS, LightningDataModule]]:
        test_dataset = self._build_dataset("test")
        if test_dataset:
            return torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.cfg.experiment.test.data.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.experiment.test.data.num_workers),
                shuffle=self.cfg.experiment.test.data.shuffle,
            )
        else:
            return None

    def exec_task(self, task: str) -> None:
        if task == 'test':
            self.test()
        else:
            super().exec_task(task)

    def train(self) -> None:
        """
        All train happens here
        """
        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if "checkpointing" in self.cfg.experiment.training:
            callbacks.append(
                ModelCheckpoint(
                    pathlib.Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]) / "checkpoints",
                    **self.cfg.experiment.training.checkpointing,
                )
            )

        gradient_clip_val = None if 'clipping' not in dir(self.cfg.experiment.training) else self.cfg.experiment.training.clipping
        trainer = pl.Trainer(
            max_epochs=self.cfg.experiment.epochs,
            accelerator='auto',
            logger=self.logger,
            devices="auto",
            strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            callbacks=callbacks,
            val_check_interval=self.cfg.experiment.validation.check_interval,
            limit_val_batches=self.cfg.experiment.validation.limit_batch,
            check_val_every_n_epoch=self.cfg.experiment.validation.check_epoch,
            accumulate_grad_batches=self.cfg.experiment.training.optim.accumulate_grad_batches,
            precision=self.cfg.experiment.training.precision,
            gradient_clip_val=gradient_clip_val
        )
        #limit_train_batches=self.cfg.experiment.training.limit_batch,

        trainer.fit(
            self.model,
            train_dataloaders=self._build_training_loader(),
            val_dataloaders=self._build_validation_loader(),
            ckpt_path=self.ckpt_path,
        )

    def test(self) -> None:
        """
        All testing happens here
        """
        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            callbacks=callbacks,
            precision=self.cfg.precision
        )

        # Only load the checkpoint if only testing. Otherwise, it will have been loaded
        # and further trained during train.
        trainer.test(
            self.model,
            dataloaders=self._build_test_loader(),
            ckpt_path=self.ckpt_path,
        )

    def _build_dataset(self, split: str) -> Optional[torch.utils.data.Dataset]:
        if split in ['training', 'test', 'validation']:
            return self.compatible_datasets[self.cfg.dataset.name](self.cfg.dataset, split=split)
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")


class ReinforcementLearningExperiment(BasePytorchExperiment, ABC):
    """
    Abstract class for a reinforcement learning experiment
    """


