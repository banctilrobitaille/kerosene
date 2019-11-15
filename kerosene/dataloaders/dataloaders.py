import torch
from enum import Enum
from torch.utils.data import Dataset, DistributedSampler

from kerosene.exceptions.exceptions import IncorrectSamplerTypeException
from kerosene.utils.devices import is_on_multiple_devices


class DataloaderType(Enum):
    DataLoader = "DataLoader"

    def __str__(self):
        return self.value


class DataLoaderFactory(object):
    def __init__(self):
        self._dataloader = {
            "DataLoader": torch.utils.data.DataLoader
        }

    def create(self, dataset, sampler: torch.utils.data.sampler.Sampler = None, params=None):
        if is_on_multiple_devices():
            assert sampler is not None, "'sampler' argument must not be None when using multiple devices."

            if not isinstance(sampler, DistributedSampler):
                raise IncorrectSamplerTypeException(
                    "'sampler' argument must be an instance of {} if using multiple devices, but got {}".format(
                        str(DistributedSampler), str(sampler.__class__)))

        return self._dataloader[str(DataloaderType.DataLoader)](dataset, sampler,
                                                                **params) if params is not None else \
            self._dataloader[str(DataloaderType.DataLoader)](dataset)
