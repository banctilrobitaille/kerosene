from typing import Union, Callable, Iterable

import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils import clip_grad_value_

from enum import Enum
import abc


class GradientClippingStrategyType(Enum):
    Value = "value"
    Norm = "norm"

    def __str__(self):
        return self.value


class GradientClippingStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def clip(self):
        raise NotImplementedError()


class GradientNormClipping(GradientClippingStrategy):

    def __init__(self, model_parameters: Union[Iterable[torch.Tensor], torch.Tensor], max_norm: float,
                 norm_type: Union[int, float]):
        """"
        Gradient norm clipping strategy.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
        """
        self._model_parameters = model_parameters
        self._max_norm = max_norm
        self._norm_type = norm_type

    def clip(self):
        clip_grad_norm_(self._model_parameters, self._max_norm, self._norm_type)


class GradientValueClipping(GradientClippingStrategy):

    def __init__(self, model_parameters: Union[Iterable[torch.Tensor], torch.Tensor], clip_value: Union[int, float]):
        """"
        Gradient value clipping strategy.

        Args:
           clip_value (float or int): maximum allowed value of the gradients. The gradients are clipped in the range
            :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
        """
        self._model_parameters = model_parameters
        self._clip_value = clip_value

    def clip(self):
        clip_grad_value_(self._model_parameters, self._clip_value)


class GradientClippingStrategyFactory(object):
    def __init__(self):
        self._clipping_strategies = {
            "value": GradientValueClipping,
            "norm": GradientNormClipping
        }

    def create(self, clipping_strategy: Union[str, GradientClippingStrategyType], model_params, params):
        return self._clipping_strategies[str(clipping_strategy)](model_params, **params) if params is not None else \
            self._clipping_strategies[str(clipping_strategy)](model_params)

    def register(self, function: str, creator: Callable):
        self._clipping_strategies[function] = creator
