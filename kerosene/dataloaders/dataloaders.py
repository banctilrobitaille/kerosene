import multiprocessing
from typing import Callable

import torch
from torch.utils.data import Dataset

from kerosene.utils.devices import on_single_device


class DataloaderFactory(object):
    def __init__(self, train_dataset: Dataset, valid_dataset: Dataset):
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._train_sampler = None
        self._valid_sampler = None

        self._samplers = {
            "default": torch.utils.data.Sampler,
            "sequentialSampler": torch.utils.data.SequentialSampler,
            "randomSampler": torch.utils.data.RandomSampler,
            "subsetRandomSampler": torch.utils.data.SubsetRandomSampler,
            "weightedRandomSampler": torch.utils.data.WeightedRandomSampler,
            "batchSampler": torch.utils.data.BatchSampler,
            "distributedSampler": torch.utils.data.DistributedSampler
        }

    def create(self, devices, num_workers, local_rank, batch_size, sampler_fn: str = "default",
               sampler_params: dict = None, shuffle: bool = True, collate_fn: Callable = None):
        if not on_single_device(devices):
            torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=local_rank,
                                                 world_size=len(devices))
            self._train_sampler = self._samplers["distributedSampler"](self._train_dataset)
            self._valid_sampler = self._samplers["distributedSampler"](self._valid_dataset)
        else:
            self._train_sampler = self._samplers[sampler_fn](self._train_dataset, **sampler_params)
            self._valid_sampler = self._samplers[sampler_fn](self._valid_sampler, **sampler_params)

        train_loader = self._create_dataloader(self._train_dataset, self._train_sampler, num_workers, batch_size,
                                               devices, shuffle, collate_fn)
        valid_loader = self._create_dataloader(self._valid_dataset, self._valid_sampler, num_workers, batch_size,
                                               devices, shuffle, collate_fn)

        return train_loader, valid_loader

    @staticmethod
    def _create_dataloader(dataset, sampler, num_workers, batch_size, devices, shuffle, collate_fn):
        return torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=False if sampler is not None else shuffle,
                                           sampler=sampler if not on_single_device(devices) else None,
                                           num_workers=num_workers if num_workers is not None else
                                           multiprocessing.cpu_count() // len(
                                               devices) if not on_single_device(
                                               devices) else multiprocessing.cpu_count(),
                                           collate_fn=collate_fn,
                                           pin_memory=torch.cuda.is_available())
