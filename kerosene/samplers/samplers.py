from typing import Union

import torch

from torch.utils import data
from enum import Enum


class SamplerType(Enum):
    Sampler = "Sampler"
    SequentialSampler = "SequentialSampler"
    RandomSampler = "RandomSampler"
    SubsetRandomSampler = "SubsetRandomSampler"
    WeightedRandomSampler = "WeightedRandomSampler"
    BatchSampler = "BatchSampler"
    DistributedSampler = "DistributedSampler"

    def __str__(self):
        return self.value


class SamplerFactory(object):

    def __init__(self):
        self._sampler = {
            "Sampler": torch.utils.data.Sampler,
            "SequentialSampler": torch.utils.data.SequentialSampler,
            "RandomSampler": torch.utils.data.RandomSampler,
            "SubsetRandomSampler": torch.utils.data.SubsetRandomSampler,
            "WeightedRandomSampler": torch.utils.data.WeightedRandomSampler,
            "BatchSampler": torch.utils.data.BatchSampler,
            "DistributedSampler": torch.utils.data.distributed.DistributedSampler
        }

    def create(self, sampler_type: Union[str, SamplerType], dataset, params):
        return self._sampler[str(sampler_type)](dataset, **params) if params is not None else self._sampler[
            str(sampler_type)](dataset)

    def register(self, function: str, creator: torch.utils.data.Sampler):
        """
        Add a new Sampler.
        Args:
           function (str): Sampler's name.
           creator (:obj:`torch.utils.data.Sampler`): A torch Sampler object wrapping the new custom sampler class.
        """
        self._sampler[function] = creator
