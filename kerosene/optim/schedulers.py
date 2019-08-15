from enum import Enum
from typing import Union

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, \
    _LRScheduler


class SchedulerType(Enum):
    ReduceLROnPlateau = "ReduceLROnPlateau"
    StepLR = "StepLR"
    MultiStepLR = "MultiStepLR"
    ExponentialLR = "ExponentialLR"
    CosineAnnealingLR = "CosineAnnealingLR"
    CyclicLR = "CyclicL"
    CosineAnnealingWarmRestarts = "CosineAnnealingWarmRestarts"


class SchedulerFactory(object):

    def __init__(self):
        self._schedulers = {
            "ReduceLROnPlateau": ReduceLROnPlateau,
            "StepLR": StepLR,
            "MultiStepLR": MultiStepLR,
            "ExponentialLR": ExponentialLR,
            "CosineAnnealingLR": CosineAnnealingLR,
        }

    def create(self, scheduler_type: Union[str, SchedulerType], params):
        return self._schedulers[str(scheduler_type)](**params)

    def register(self, function: str, creator: _LRScheduler):
        """
        Add a new activation layer.
        Args:
           function (str): Activation layer name.
           creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom optimizer function.
        """
        self._schedulers[function] = creator
