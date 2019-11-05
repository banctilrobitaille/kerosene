import unittest

import numpy as np
import torch
from torch import nn
from hamcrest import *

from kerosene.training.gradient_clipping_mixin import GradientClippingMixin


class TestGradientClipping(unittest.TestCase):
    value_clipping_params = {"clip_value": 1.0}
    norm_clipping_params = {"max_norm": 1.0, "norm_type": 2}

    def setUp(self) -> None:
        self._mixin = GradientClippingMixin()
        self._convolution = nn.Conv2d(32, 32, 3, 1, "valid")
        self._convolution.zero_grad()
        self._convolution.weight.grad = torch.Tensor().new_full((32, 32, 3, 3), 5)

    def test_should_clip_gradients_with_norm(self):
        self._mixin.clip_gradients(self._convolution.parameters(), "norm", self.norm_clipping_params)
        norm = self._convolution.weight.grad.norm(2)
        np.testing.assert_almost_equal(norm.item(), 1.0, decimal=4)

    def test_should_clip_gradients_with_value(self):
        self._mixin.clip_gradients(self._convolution.parameters(), "value", self.value_clipping_params)
        assert_that(
            np.less_equal(self._convolution.weight.grad.numpy(),
                          np.full((32, 32, 3, 3), self.value_clipping_params["clip_value"])).any(),
            is_(True))
        assert_that(
            np.greater_equal(self._convolution.weight.grad.numpy(),
                             np.full((32, 32, 3, 3), -self.value_clipping_params["clip_value"])).any(),
        )

    def test_should_not_clip_gradients_with_None_func(self):
        self._mixin.clip_gradients(self._convolution.parameters(), None, self.value_clipping_params)
        np.testing.assert_array_equal(self._convolution.weight.grad.numpy(), np.full((32, 32, 3, 3), 5))
