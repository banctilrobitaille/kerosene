from abc import ABC
from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from kerosene.config.trainers import RunConfiguration
from kerosene.utils.distributed import on_single_device

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


class ApexLoss(object):
    def __init__(self, loss_id: int, loss: Tensor, optimizer: Optimizer):
        super().__init__()
        self._loss_id = loss_id
        self._loss = loss
        self._optimizer = optimizer

    @property
    def loss_id(self):
        return self._loss_id

    @property
    def loss(self):
        return self._loss

    def backward(self, gradient: Optional[Tensor] = None, retain_graph=False,
                 create_graph=False) -> None:
        if APEX_AVAILABLE:
            with amp.scale_loss(self._loss, self._optimizer, loss_id=self._loss_id) as scaled_loss:
                scaled_loss.backward(gradient, retain_graph, create_graph)
        else:
            self._loss.backward(gradient, retain_graph, create_graph)

    def __add__(self, other):
        if isinstance(other, Tensor):
            self._loss = self._loss + other
        elif isinstance(other, ApexLoss):
            self._loss = self._loss + other.loss
        else:
            raise NotImplementedError("Cannot add an element of type: {} to an ApexLoss.".format(str(type(other))))
        return self

    def __mul__(self, value: Union[int, float]):
        self._loss = self._loss * value
        return self

    def __rmul__(self, value: Union[int, float]):
        self._loss = self._loss * value
        return self

    def __eq__(self, other):
        if isinstance(other, Tensor):
            is_equal = torch.all(torch.eq(self._loss, other))
        elif isinstance(other, ApexLoss):
            is_equal = torch.all(torch.eq(self._loss, other.loss))
        else:
            raise NotImplementedError("Cannot compare an element of type: {} to an ApexLoss.".format(str(type(other))))
        return is_equal


class ApexModule(ABC, nn.Module):
    def __init__(self, model, optimizer, amp_id=0, use_amp=True):
        super().__init__()
        self._amp_id = amp_id
        self._use_amp = use_amp
        self._model = model
        self._optimizer = optimizer

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def amp_id(self):
        return self._amp_id

    @property
    def use_amp(self):
        return self._use_amp

    def initialize(self, amp_id: int, num_losses: int, run_config: RunConfiguration):
        self._amp_id = amp_id
        self._use_amp = run_config.use_amp
        torch.cuda.set_device(run_config.devices[run_config.local_rank])
        self._model.to(run_config.devices[run_config.local_rank])

        if APEX_AVAILABLE and self._use_amp:
            self._model, self._optimizer = amp.initialize(
                self._model, self._optimizer, opt_level=run_config.amp_opt_level, num_losses=num_losses)

        if not on_single_device(run_config.devices):
            self._model = DDP(self._model, delay_allreduce=True)
