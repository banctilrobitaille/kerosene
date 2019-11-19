import unittest

import torch
from hamcrest import *
from torch import nn

from kerosene.nn.utils.gradients import GradientClippingStrategyFactory, GradientClippingStrategyType, \
    GradientClippingStrategy


class TestGradientClippingStrategyFactory(unittest.TestCase):
    def setUp(self) -> None:
        self._model = nn.Conv2d(32, 32, 3, 1, "valid")
        self._optim = torch.optim.SGD(self._model.parameters(), 0.1)

    def test_should_return_a_gradient_norm_clipping_strategy(self):
        strategy = GradientClippingStrategyFactory().create(GradientClippingStrategyType.Norm, self._model.parameters(),
                                                            {"max_norm": 1.0, "norm_type": 2})

        assert_that(strategy, instance_of(GradientClippingStrategy))

    def test_should_return_a_gradient_value_clipping_strategy(self):
        strategy = GradientClippingStrategyFactory().create(GradientClippingStrategyType.Value,
                                                            self._model.parameters(), {"clip_value": 1.0})

        assert_that(strategy, instance_of(GradientClippingStrategy))
