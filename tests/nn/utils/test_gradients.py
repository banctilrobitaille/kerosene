import unittest

from hamcrest import *

from kerosene.nn.utils.gradients import GradientClippingStrategyFactory, GradientClippingStrategyType, \
    GradientClippingStrategy


class TestGradientClippingStrategyFactory(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_should_return_a_gradient_norm_clipping_strategy(self):
        strategy = GradientClippingStrategyFactory().create(GradientClippingStrategyType.Norm,
                                                            {"max_norm": 1.0, "norm_type": 2})

        assert_that(strategy, instance_of(GradientClippingStrategy))

    def test_should_return_a_gradient_value_clipping_strategy(self):
        strategy = GradientClippingStrategyFactory().create(GradientClippingStrategyType.Value, {"clip_value": 1.0})

        assert_that(strategy, instance_of(GradientClippingStrategy))
