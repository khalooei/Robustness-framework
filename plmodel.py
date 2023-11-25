'''
Developted by : Mohammad Khalooei (Khalooei@aut.ac.ir)

In this file, you can follow the main pytorch lightning module which covers all the training procedures based on pytorch lightning library.
You can see that `PLModel` class covers all functions as we need.

## Light tips in Pytorch lightning : 
# 1- model
# 2- optimizer
# 3- data
# 4- training loop "the magic"
# 5- validation loop "the validation magic"

'''


from typing import Any, List, Optional, Union
# from lightning_fabric.utilities.types import Steppable
# from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

import time
import pathlib
import logging
import os
import time
import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
from torch import Tensor, optim, nn
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.core.saving import save_hparams_to_yaml
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities import rank_zero_only

from torch.optim.lr_scheduler import OneCycleLR
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

import hydra
from omegaconf import DictConfig, OmegaConf

import torchmetrics
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

from autoattack import AutoAttack
from torchattacks import PGD, FGSM, FFGSM, APGD, TPGD, CW,MIFGSM,DeepFool
from architectures.MKToyNet import MKToyNet
from architectures.WideResNet import WideResNet16
from architectures.ResNet import ResNet18,ResNet50,ResNet18_benchmark
from architectures.LeNet import LeNet

from utils import *
from lightning.pytorch.accelerators import find_usable_cuda_devices

from torch.optim.swa_utils import AveragedModel, update_bn

class PLModel(pl.LightningModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        if self.cfg.global_params.architecture == "MKToyNet":
            self.model = MKToyNet(input_size=28)
        elif self.cfg.global_params.architecture == "LeNet":
            self.model = LeNet()
        elif self.cfg.global_params.architecture == "WideResNet":
            self.model = WideResNet16()
        elif self.cfg.global_params.architecture == "ResNet":
            #self.model = ResNet18()
            self.model = ResNet18_benchmark()
        
        self.loss = self.model.loss
        self.metric_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10) #accuracy(preds, target)
            
        self.validation_step_outputs = {}
        self.validation_step_outputs['clean_val_loss']=[]
        self.validation_step_outputs['clean_val_acc']=[]

        
        # weight average idea
        self.swa_model = AveragedModel(self)
        # Important: This property activates manual optimization.
        # self.automatic_optimization = False

        # save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters()

        # self.save_hyperparameters(ignore=['model']) 



    def set_cfg(self, cfg):
        self.cfg=cfg

    def forward(self, x):
        logits = self.model(x)
        return logits
    
    def configure_optimizers(self):
        # optimizer  if we not mentioned here, we must pass it in the Trainer 
        # optimizer = optim.SGD(self.parameters(),lr=1e-2)
        
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.training_params.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = len(self.train_idx) #// self.cfg.training_params.batch_size
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
    
    @torch.enable_grad()
    def prepare_adversarial_data(self, batch, attack):
        x,y = batch
        # with torch.inference_mode():
        x_adv = attack(x,y)
        return (x, x_adv, y)


    def _adversarial_at_training_step(self,batch,attack):
        x, x_adv, y = self.prepare_adversarial_data(batch, attack)
        # 1- forward
        logits = self(x_adv)
        # 2- loss
        J = self.loss(logits, y)  

        return J, y, torch.argmax(logits, dim=1)
    
    # @torch.enable_grad()
    def _shared_step(self, batch):
        x,y = batch
        # 1- forward
        logits = self(x)
        # 2- loss
        J = self.loss(logits, y)  # ce loss
        return J, y, torch.argmax(logits, dim=1)
    
    # the core function :) of training
    def training_step(self, batch, batch_step):
        # get batch + feed to model and calculate loss
         # batch = on_batch_start(batch) :)
        # opt = self.optimizers()
        # opt.zero_grad()

        if self.cfg.training_params.type == 'normal':
            loss, true_labels, predicted_labels = self._shared_step(batch)
        elif self.cfg.training_params.type == 'AT':
            if self.cfg.adversarial_training_params.name == 'PGD':
                atk = PGD(self, eps=self.cfg.adversarial_training_params.eps, steps=10, random_start=True)
            elif self.cfg.adversarial_training_params.name == 'FGSM':
                atk = FGSM(self, eps=self.cfg.adversarial_training_params.eps)
            elif self.cfg.adversarial_training_params.name == 'Fast':
                atk = FFGSM(self, eps=self.cfg.adversarial_training_params.eps)
            elif self.cfg.adversarial_training_params.name == 'TPGD':
                atk = FFGSM(self, eps=self.cfg.adversarial_training_params.eps)
            elif self.cfg.adversarial_training_params.name == 'MIFGSM':
                atk = MIFGSM(self, eps=self.cfg.adversarial_training_params.eps)
            elif self.cfg.adversarial_training_params.name == 'CW':
                atk = CW(self)
            else:
                atk = PGD(self, eps=self.cfg.adversarial_training_params.eps, steps=10, random_start=True)
           
            loss, true_labels, predicted_labels = self._adversarial_at_training_step(batch, atk)

        # self.manual_backward(loss)
        # opt.step()

        acc = self.metric_acc(predicted_labels,true_labels)
        self.log("clean_train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("clean_train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'clean_train_acc':acc} # for log
    
    # on_after_backward function :)
    # def backward(self, trainer, loss, optimizer):
    #     loss.backward()

    def training_epoch_end(self, training_step_outputs):
        self.swa_model.update_parameters(self.model)
        
    def on_train_end(self):
        update_bn(self.trainer.train_dataloader, self.swa_model, device=self.device)

    
    def validation_step(self, batch, batch_idx):
        # results = self.normal_training_params(batch, batch_idx)
        results = {}
        loss, true_labels, predicted_labels = self._shared_step(batch)
        acc = self.metric_acc(predicted_labels, true_labels)
        self.validation_step_outputs['clean_val_loss'].append(loss)  # for hook
        self.validation_step_outputs['clean_val_acc'].append(acc)  # for hook
        
        results['clean_val_acc']=acc
        results['clean_val_loss']=loss
        # del results['train_acc']
        return results
    
    def on_validation_epoch_end(self): #validation_epoch_end(self, val_step_outputs):
        # [results batch1, results batch 2, ..]
        val_loss = torch.stack(self.validation_step_outputs['clean_val_loss'])
        val_acc = torch.stack(self.validation_step_outputs['clean_val_acc'])
        # do something with all preds
        ...
        avg_val_loss = torch.Tensor([x for x in val_loss]).mean()
        avg_val_acc = torch.Tensor([x for x in val_acc]).mean()

        self.validation_step_outputs.clear()  # free memory
        self.validation_step_outputs = {}
        self.validation_step_outputs['clean_val_loss']=[]
        self.validation_step_outputs['clean_val_acc']=[]
        
        self.log("clean_val_acc", avg_val_acc, prog_bar=True)
        self.log("clean_val_loss", avg_val_loss, prog_bar=True)
        return {'clean_val_loss': avg_val_loss} #early stopping for val_loss just field.
    

    # @torch.enable_grad
    # def on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):


    @torch.inference_mode(False)
    def test_step(self, batch, batch_idx): # not executed during the training
        x,y = batch

        # 1- forward
        logits = self(x)
        true_labels, predicted_labels = y,torch.argmax(logits, dim=1)
        test_acc = self.metric_acc(predicted_labels, true_labels)
        self.log('test_clean_accuracy',test_acc)

        # attack_name = "CW"
        # atk = CW(self)
        # x, x_adv, y = self.prepare_adversarial_data(batch, atk)
        # logits = self(x_adv)
        # true_labels, predicted_labels = y,torch.argmax(logits, dim=1)
        # test_acc = self.metric_acc(predicted_labels, true_labels)
        # self.log(f'test_adv-{attack_name}_accuracy',test_acc)

        # attack_name = "DeepFool"
        # atk = DeepFool(self)
        # x, x_adv, y = self.prepare_adversarial_data(batch, atk)
        # logits = self(x_adv)
        # true_labels, predicted_labels = y,torch.argmax(logits, dim=1)
        # test_acc = self.metric_acc(predicted_labels, true_labels)
        # self.log(f'test_adv-{attack_name}_accuracy',test_acc)

        attack_name = "PGD"
        for eps in [0.01,0.03, 0.1,0.2,0.3,0.5]:
            # atk = PGD(self, eps=self.cfg.adversarial_training_params.eps, steps=10, random_start=True)
            atk = PGD(self, eps=eps, steps=50, random_start=True)
            x, x_adv, y = self.prepare_adversarial_data(batch, atk)
            logits = self(x_adv)
            true_labels, predicted_labels = y,torch.argmax(logits, dim=1)
            test_acc = self.metric_acc(predicted_labels, true_labels)
            self.log(f'test_adv-{attack_name}-eps{eps}_accuracy',test_acc)

        # if self.current_epoch >150:
        attack_name = "FGSM"
        for eps in [0.01,0.03, 0.1,0.2,0.3,0.5]:
            # atk = PGD(self, eps=self.cfg.adversarial_training_params.eps, steps=10, random_start=True)
            atk = FGSM(self, eps=eps)
            x, x_adv, y = self.prepare_adversarial_data(batch, atk)
            logits = self(x_adv)
            true_labels, predicted_labels = y,torch.argmax(logits, dim=1)
            test_acc = self.metric_acc(predicted_labels, true_labels)
            self.log(f'test_adv-{attack_name}-eps{eps}_accuracy',test_acc)
        
        # # if self.current_epoch >150:
        # attack_name = "MIFGSM"
        # for eps in [0.01,0.03, 0.1,0.2,0.3,0.5]:
        #     # atk = PGD(self, eps=self.cfg.adversarial_training_params.eps, steps=10, random_start=True)
        #     atk = MIFGSM(self, eps=eps, alpha=0.0392156862745098)
        #     x, x_adv, y = self.prepare_adversarial_data(batch, atk)
        #     logits = self(x_adv)
        #     true_labels, predicted_labels = y,torch.argmax(logits, dim=1)
        #     test_acc = self.metric_acc(predicted_labels, true_labels)
        #     self.log(f'test_adv-{attack_name}-eps{eps}_accuracy',test_acc)
        
        # attack_name = "APGD"
        # for eps in [0.03, 0.1,0.2,0.3,0.5]:
        #     # atk = PGD(self, eps=self.cfg.adversarial_training_params.eps, steps=10, random_start=True)
        #     # atk = APGD(self, eps=eps)
        #     adversary = AutoAttack(self, norm='Linf', eps=eps, version='standard')
        #     x_adv = adversary.run_standard_evaluation(x, y, bs=len(x))

        #     # x, x_adv, y = self.prepare_adversarial_data(batch, atk)
        #     logits = self(x_adv)
        #     true_labels, predicted_labels = y,torch.argmax(logits, dim=1)
        #     test_acc = self.metric_acc(predicted_labels, true_labels)
        #     self.log(f'test_adv-{attack_name}-eps{eps}_accuracy',test_acc)
            

    def adversarial_test_step(self, batch, batch_idx): # not executed during the training
        loss, true_labels, predicted_labels = self._shared_step(batch)
        test_acc = self.metric_acc(predicted_labels, true_labels)
        self.log('test_accuracy',test_acc)

    # Data loader -> if we not mentioned here, we must pass it in the Trainer 
    def prepare_data(self):
        # one time is being run (lazy loading)   #it is useful for multi-GPU
        if self.cfg.global_params.dataset == 'MNIST':
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            )
            test_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            )
            train_data = datasets.MNIST('data', train=True, download=True, transform=train_transforms)
            test_data = datasets.MNIST('data', train=False, download=True, transform=test_transforms)
            return train_data, test_data
        elif self.cfg.global_params.dataset == 'CIFAR10':
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomCrop(32, padding=4),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    # cifar10_normalization(),
                ]
            )
            test_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    # cifar10_normalization(),
                ]
            )
            train_data = datasets.CIFAR10('data', train=True, download=True, transform=train_transforms)
            test_data = datasets.CIFAR10('data', train=False, download=True, transform=test_transforms)
            return train_data, test_data
        else:
            print('Dataset Error')
            exit(-1)

    def setup(self, stage):
        # any transformation here ....
        if self.cfg.global_params.dataset == 'MNIST':
            # train_data = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
            train_data, test_data = self.prepare_data()
            self.train_idx, self.val_idx = random_split(train_data, [55000,5000])
            self.test_data = test_data
            # self.test_data = datasets.MNIST('data', train=False, download=False, transform=transforms.ToTensor())
            # self.train_idx, self.val_idx = random_split(test_data, [55000,5000])
        elif self.cfg.global_params.dataset == 'CIFAR10':
            #train_data = datasets.CIFAR10('data', train=True, download=False, transform=transforms.ToTensor())
            train_data, test_data = self.prepare_data()
            self.train_idx, self.val_idx = random_split(train_data, [len(train_data)-5000,5000])
            self.test_data = test_data
            #self.test_data = datasets.CIFAR10('data', train=False, download=False, transform=transforms.ToTensor())
        
    def train_dataloader(self):
        # train / val split
        self.train_loader = DataLoader(self.train_idx, batch_size=self.cfg.training_params.batch_size,num_workers=self.cfg.training_params.dataloader_workers,pin_memory=True)
        return self.train_loader
    
    def val_dataloader(self):
        self.val_loader = DataLoader(self.val_idx, batch_size=self.cfg.training_params.batch_size,num_workers=self.cfg.training_params.dataloader_workers,pin_memory=True)
        return self.val_loader
    
    def test_dataloader(self):
        self.test_loader = DataLoader(self.test_data , batch_size=self.cfg.training_params.batch_size,num_workers=self.cfg.training_params.dataloader_workers,pin_memory=True)
        return self.test_loader
