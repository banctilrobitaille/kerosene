import multiprocessing
from typing import Callable

import torch
from torch.utils.data import Dataset

from kerosene.config.trainers import RunConfiguration, TrainerConfiguration
from kerosene.utils.devices import on_single_device


class DataloaderFactory(object):
    def __init__(self, train_dataset: Dataset, valid_dataset: Dataset, test_dataset: Dataset):
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset

    def create(self, run_config: RunConfiguration, training_config: TrainerConfiguration, collate_fn: Callable = None):
        devices = run_config.devices
        train_sampler = None
        valid_sampler = None
        test_sampler = None

        if not on_single_device(devices):
            train_sampler = torch.utils.data.distributed.DistributedSampler(self._train_dataset)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(self._valid_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(self._test_dataset)

        train_loader = self._create_dataloader(train_sampler, run_config, training_config, devices, collate_fn)
        valid_loader = self._create_dataloader(valid_sampler, run_config, training_config, devices, collate_fn)
        test_loader = self._create_dataloader(test_sampler, run_config, training_config, devices, collate_fn)

        return train_loader, valid_loader, test_loader

    def _create_dataloader(self, sampler, run_config, training_config, devices, collate_fn):
        return torch.utils.data.DataLoader(dataset=self._train_dataset,
                                           batch_size=training_config.batch_size,
                                           shuffle=False if not on_single_device(devices) else True,
                                           num_workers=run_config.num_workers if run_config.num_workers is not None else
                                           multiprocessing.cpu_count() // len(
                                               run_config.devices) if not on_single_device(
                                               devices) else multiprocessing.cpu_count(),
                                           sampler=sampler if not on_single_device(devices) else None,
                                           collate_fn=collate_fn,
                                           pin_memory=torch.cuda.is_available())
