import unittest

import torch
from kerosene.optim.optimizers import OptimizerFactory
from hamcrest import *


class OptimizerFactoryTest(unittest.TestCase):

    def setUp(self) -> None:
        self._valid_optimizer_config = [
            {"name": "weight", "type": "SGD",
             "params": {"lr": 0.01, "momentum": 0.5, "weight_decay": 0.0, "model_parameters": "weight"}},
            {"name": "bias", "type": "SGD",
             "params": {"lr": 0.01, "momentum": 0.5, "weight_decay": 0.0, "model_parameters": "bias"}}]
        self._model = torch.nn.Conv2d(3, 32, (3, 3), (1, 1), bias=True)

    def test_should_create_optimizer_with_provided_config(self):
        optimizer_weights = OptimizerFactory().create(self._valid_optimizer_config[0]["type"], self._model,
                                                      **self._valid_optimizer_config[0]["params"])
        optimizer_bias = OptimizerFactory().create(self._valid_optimizer_config[1]["type"], self._model,
                                                   **self._valid_optimizer_config[1]["params"])

        num_bias = sum(p.numel() for n, p in self._model.named_parameters() if p.requires_grad and "bias" in n)
        num_weights = sum(p.numel() for n, p in self._model.named_parameters() if p.requires_grad and "weight" in n)

        assert_that(optimizer_bias, instance_of(torch.optim.Optimizer))
        assert_that(optimizer_weights, instance_of(torch.optim.Optimizer))

        assert_that(optimizer_bias.param_groups[0]["params"][0].flatten().shape, is_(torch.Size([num_bias])))
        assert_that(optimizer_weights.param_groups[0]["params"][0].flatten().shape, is_(torch.Size([num_weights])))
