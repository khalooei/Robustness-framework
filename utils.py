from typing import Any, List, Optional, Union

import time
import pathlib
import logging
import os
import time
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import Tensor, optim, nn
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.core.saving import save_hparams_to_yaml
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities import rank_zero_only

import hydra
from omegaconf import DictConfig, OmegaConf

import torchmetrics
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

from torchattacks import PGD, FGSM, FFGSM, APGD, TPGD, CW
from architectures.MKToyNet import MKToyNet
from architectures.WideResNet import WideResNet16
from architectures.ResNet import ResNet18,ResNet50
from architectures.LeNet import LeNet

torch.manual_seed(313) # reproducibility
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

class KhalooeiLoggingLogger(Logger):
    def __init__(self,save_dir='logs/',version=f"{time.strftime('%Y%m%d%H%M%S')}"):
        # Create a 'logs' directory if it doesn't exist
        s_output_dir = os.path.join(save_dir,'lightning_logs',version)
        os.makedirs(s_output_dir, exist_ok=True)

        # Include a timestamp in the log file name
        experiment_name = f"experiment_{version}.log"

        # Configure logging with the experiment-specific log file
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s]: %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(s_output_dir, experiment_name)),
                logging.StreamHandler(),
            ]
        )

        self.logger = logging.getLogger(__name__)

    @property
    def name(self):
        return "KhalooeiLoggingLogger"
    
    @property
    def version(self):
        # Return the experiment version, int or str.
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        # Log metrics to the console and log file
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value}")

    @rank_zero_only
    def log_hyperparams(self, params):
        # Log hyperparameters to the console and log file
        self.logger.info("Hyperparameters:")
        for key, value in params.items():
            self.logger.info(f"{key}: {value}")

    @rank_zero_only
    def experiment(self):
        # Return the experiment object if available (required by the BaseLogger)
        return None
    @rank_zero_only
    def save(self):
        # Return the experiment object if available (required by the BaseLogger)
        pass
    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass

class CustomTimeCallback(Callback):
    def on_train_start(self, trainer, lightning_module):
        self.start = time.time()
        print("Training is starting ...")
    def on_train_end(self, trainer, lightning_module):
        self.end = time.time()
        total_miniutes = (self.end-self.start)/60
        print(f"Training is finished. It took {total_miniutes} min.")
