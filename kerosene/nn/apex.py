# -*- coding: utf-8 -*-
# Copyright 2019 Kerosene Authors. All Rights Reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import division

from abc import ABC
from typing import Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from kerosene.config.trainers import RunConfiguration
from kerosene.utils.devices import on_single_device

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

    def cpu(self):
        return ApexLoss(self._loss_id, self._loss.cpu(), self._optimizer)

    def detach(self):
        return ApexLoss(self._loss_id, self._loss.detach(), self._optimizer)

    def float(self):
        loss = self._loss.float()
        transformed_loss = ApexLoss(self._loss_id, loss, self._optimizer)
        return transformed_loss

    def numpy(self):
        return self._loss.numpy()

    def mean(self):
        return ApexLoss(self._loss_id, torch.mean(self._loss), self._optimizer)

    def __add__(self, other):
        if isinstance(other, (Tensor, int, float)):
            loss = ApexLoss(self._loss_id, torch.add(self._loss, other), self._optimizer)
        elif isinstance(other, ApexLoss):
            loss = ApexLoss(self._loss_id, torch.add(self._loss, other.loss), self._optimizer)
        else:
            raise NotImplementedError("Cannot add an element of type: {} to an ApexLoss.".format(str(type(other))))
        return loss

    def __mul__(self, other):
        if isinstance(other, (Tensor, int, float)):
            loss = ApexLoss(self._loss_id, torch.mul(self._loss, other), self._optimizer)
        elif isinstance(other, ApexLoss):
            loss = ApexLoss(self._loss_id, torch.mul(self._loss, other.loss), self._optimizer)
        else:
            raise NotImplementedError("Cannot mul an element of type: {} to an ApexLoss.".format(str(type(other))))
        return loss

    def __rmul__(self, other):
        if isinstance(other, (Tensor, int, float)):
            loss = ApexLoss(self._loss_id, torch.mul(self._loss, other), self._optimizer)
        else:
            raise NotImplementedError("Cannot rmul an element of type: {} to an ApexLoss.".format(str(type(other))))
        return loss

    def __truediv__(self, other):
        if isinstance(other, (Tensor, int, float)):
            loss = ApexLoss(self._loss_id, torch.div(self._loss, other), self._optimizer)
        elif isinstance(other, ApexLoss):
            loss = ApexLoss(self._loss_id, torch.div(self._loss, other.loss), self._optimizer)
        else:
            raise NotImplementedError("Cannot truediv an element of type: {} to an ApexLoss.".format(str(type(other))))
        return loss

    def __rtruediv__(self, other):
        if isinstance(other, (Tensor, int, float)):
            loss = ApexLoss(self._loss_id, torch.div(other, self._loss), self._optimizer)
        else:
            raise NotImplementedError("Cannot rtruediv an element of type: {} to an ApexLoss.".format(str(type(other))))
        return loss

    def __sub__(self, other):
        if isinstance(other, (Tensor, int, float)):
            loss = ApexLoss(self._loss_id, torch.sub(self._loss, other), self._optimizer)
        elif isinstance(other, ApexLoss):
            loss = ApexLoss(self._loss_id, torch.sub(self._loss, other.loss), self._optimizer)
        else:
            raise NotImplementedError(
                "Cannot substract an element of type: {} to an ApexLoss.".format(str(type(other))))
        return loss

    def __rsub__(self, other):
        if isinstance(other, (Tensor, int, float)):
            loss = ApexLoss(self._loss_id, torch.sub(other, self._loss), self._optimizer)
        else:
            raise NotImplementedError(
                "Cannot substract an element of type: {} to an ApexLoss.".format(str(type(other))))
        return loss

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
        self._model.to(run_config.device)

        if APEX_AVAILABLE and self._use_amp:
            self._model, self._optimizer = amp.initialize(
                self._model, self._optimizer, opt_level=run_config.amp_opt_level, num_losses=num_losses)
        if not on_single_device(run_config.devices):
            self._model = DDP(self._model, delay_allreduce=True)
