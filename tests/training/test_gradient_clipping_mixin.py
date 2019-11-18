import unittest

import numpy as np
import torch

from kerosene.training import Status
from kerosene.training.trainers import ModelTrainer
from torch import nn
from hamcrest import *
from mockito import mock
from kerosene.training.gradient_clipping_mixin import GradientClippingMixin
from kerosene.metrics.metrics import Metric


class TestGradientClipping(unittest.TestCase):
    value_clipping_params = {"clip_value": 1.0}
    norm_clipping_params = {"max_norm": 1.0, "norm_type": 2}

    def setUp(self) -> None:
        self._model = nn.Conv2d(32, 32, 3, 1, "valid")
        self._optim = torch.optim.SGD(self._model.parameters(), 0.1)

    def test_should_clip_gradients_with_norm(self):
        model_trainer = ModelTrainer("test model",
                                     self._model,
                                     mock(nn.Module),
                                     self._optim,
                                     mock(torch.optim.lr_scheduler.StepLR),
                                     mock(Metric),
                                     gradient_clipping_func="norm",
                                     gradient_clipping_params=self.norm_clipping_params)
        model_trainer._status = Status.TRAIN
        model_trainer.zero_grad()
        model_trainer._model.weight.grad = torch.Tensor().new_full((32, 32, 3, 3), 5)
        model_trainer.step()
        norm = model_trainer.model.weight.grad.norm(2)
        np.testing.assert_almost_equal(norm.item(), 1.0, decimal=4)

    def test_should_clip_gradients_with_value(self):
        model_trainer = ModelTrainer("test model",
                                     self._model,
                                     mock(nn.Module),
                                     self._optim,
                                     mock(torch.optim.lr_scheduler.StepLR),
                                     mock(Metric),
                                     gradient_clipping_func="value",
                                     gradient_clipping_params=self.value_clipping_params)
        model_trainer._status = Status.TRAIN
        model_trainer.zero_grad()
        model_trainer._model.weight.grad = torch.Tensor().new_full((32, 32, 3, 3), 5)
        model_trainer.step()
        assert_that(
            np.less_equal(model_trainer.model.weight.grad.numpy(),
                          np.full((32, 32, 3, 3), self.value_clipping_params["clip_value"])).any(),
            is_(True))
        assert_that(
            np.greater_equal(model_trainer.model.weight.grad.numpy(),
                             np.full((32, 32, 3, 3), -self.value_clipping_params["clip_value"])).any(),
        )

    def test_should_not_clip_gradients_with_None_func(self):
        model_trainer = ModelTrainer("test model",
                                     self._model,
                                     mock(nn.Module),
                                     self._optim,
                                     mock(torch.optim.lr_scheduler.StepLR),
                                     mock(Metric))
        model_trainer._status = Status.TRAIN
        model_trainer.zero_grad()
        model_trainer._model.weight.grad = torch.Tensor().new_full((32, 32, 3, 3), 5)
        model_trainer.step()
        np.testing.assert_array_equal(model_trainer.model.weight.grad.numpy(), np.full((32, 32, 3, 3), 5))
