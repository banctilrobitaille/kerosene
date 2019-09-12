import multiprocessing
from typing import Callable

import torch
from torch.utils.data import Dataset

from kerosene.config.trainers import RunConfiguration, TrainerConfiguration
from kerosene.utils.devices import on_single_device


class DataloaderFactory(object):
    def __init__(self, train_dataset: Dataset, valid_dataset: Dataset):
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset

    def create(self, run_config: RunConfiguration, training_config: TrainerConfiguration, collate_fn: Callable = None):
        devices = run_config.devices
        if not on_single_device(devices):
            torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=run_config.local_rank)
            train_sampler = torch.utils.data.distributed.DistributedSampler(self._train_dataset)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(self._valid_dataset)

        train_loader = torch.utils.data.DataLoader(dataset=self._train_dataset,
                                                   batch_size=training_config.batch_size,
                                                   shuffle=False if not on_single_device(devices) else True,
                                                   num_workers=multiprocessing.cpu_count() // 2 if not on_single_device(
                                                       devices) else multiprocessing.cpu_count(),
                                                   sampler=train_sampler if not on_single_device(devices) else None,
                                                   collate_fn=collate_fn,
                                                   pin_memory=torch.cuda.is_available())

        valid_loader = torch.utils.data.DataLoader(dataset=self._valid_dataset,
                                                   batch_size=training_config.batch_size,
                                                   shuffle=False if not on_single_device(devices) else True,
                                                   num_workers=multiprocessing.cpu_count() // 2 if not on_single_device(
                                                       devices) else multiprocessing.cpu_count(),
                                                   sampler=valid_sampler if not on_single_device(devices) else None,
                                                   collate_fn=collate_fn,
                                                   pin_memory=torch.cuda.is_available())
        return train_loader, valid_loader
