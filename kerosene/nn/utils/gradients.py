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

import abc
from enum import Enum
from typing import Union, Callable, Iterable

import torch
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


class GradientClippingStrategyType(Enum):
    Value = "value"
    Norm = "norm"

    def __str__(self):
        return self.value


class GradientClippingStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def clip(self, model_parameters: Union[Iterable[torch.Tensor], torch.Tensor]):
        raise NotImplementedError()


class GradientNormClipping(GradientClippingStrategy):

    def __init__(self, max_norm: float, norm_type: Union[int, float] = 2):
        """"
        Gradient norm clipping strategy.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
        """
        self._max_norm = max_norm
        self._norm_type = norm_type

    def clip(self, model_parameters):
        clip_grad_norm_(model_parameters, self._max_norm, self._norm_type)


class GradientValueClipping(GradientClippingStrategy):

    def __init__(self, clip_value: Union[int, float]):
        """"
        Gradient value clipping strategy.

        Args:
           clip_value (float or int): maximum allowed value of the gradients. The gradients are clipped in the range
            :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
        """
        self._clip_value = clip_value

    def clip(self, model_parameters: Union[Iterable[torch.Tensor], torch.Tensor]):
        clip_grad_value_(model_parameters, self._clip_value)


class GradientClippingStrategyFactory(object):
    def __init__(self):
        self._clipping_strategies = {
            "value": GradientValueClipping,
            "norm": GradientNormClipping
        }

    def create(self, clipping_strategy: Union[str, GradientClippingStrategyType], params):
        if clipping_strategy is not None:
            return self._clipping_strategies[str(clipping_strategy)](**params)
        else:
            return None

    def register(self, function: str, creator: Callable):
        self._clipping_strategies[function] = creator
