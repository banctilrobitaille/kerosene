from abc import ABC
from typing import Optional

from torch import Tensor, nn
from torch.optim import Optimizer

from kerosene.config.trainers import RunConfiguration
from kerosene.utils.distributed import on_single_device

try:
    from apex import amp

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

    def backward(self, gradient: Optional[Tensor] = None, keep_graph=False,
                 create_graph=False) -> None:
        if APEX_AVAILABLE:
            with amp.scale_loss(self._loss, self._optimizer, loss_id=self._loss_id) as scaled_loss:
                scaled_loss.backward(gradient, keep_graph, create_graph)
        else:
            self._loss.backward(gradient, keep_graph, create_graph)


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

        if on_single_device(run_config.devices):
            self._model.cuda(device=run_config.devices[0])
        else:
            # TODO implement distributed training
            raise NotImplementedError("Distributed training is not yet supported")

        if APEX_AVAILABLE and run_config.use_amp:
            self._model, self._optimizer = amp.initialize(
                self._model, self._optimizer, opt_level=run_config.amp_opt_level, num_losses=num_losses)
