import os
import unittest

import torch
from hamcrest import *
from kerosene.config.trainers import RunConfiguration

from kerosene.nn.apex import ApexLoss, ApexModule
from kerosene.optim.optimizers import OptimizerFactory


class TestApexLoss(unittest.TestCase):
    VALID_LOSS_VALUE = torch.tensor([1.0, 2.0, 3.0])
    INVALID_LOSS_VALUE = "Ford Fiesta"

    VALID_SCALAR = 2.0
    INVALID_SCALAR = "Ford Focus"

    VALID_LOSS_ID = 0
    INVALID_LOSS_ID = "Ford F150"

    def test_should_add_losses(self):
        expected_result = ApexLoss(self.VALID_LOSS_ID, torch.tensor([2.0, 4.0, 6.0]), None)

        apex_loss1 = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE, None)
        apex_loss2 = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE, None)

        assert_that(apex_loss1 + apex_loss2, equal_to(expected_result))

    def test_should_multiply_by_a_scalar(self):
        expected_result = ApexLoss(self.VALID_LOSS_ID, self.VALID_SCALAR * self.VALID_LOSS_VALUE, None)

        loss = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE, None)

        assert_that(self.VALID_SCALAR * loss, equal_to(expected_result))
        assert_that(loss * self.VALID_SCALAR, equal_to(expected_result))

    # noinspection PyTypeChecker
    def test_should_divide_by_a_scalar(self):
        left_div_expected_result = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE / self.VALID_SCALAR,
                                            None)
        right__div_expected_result = ApexLoss(self.VALID_LOSS_ID, self.VALID_SCALAR / self.VALID_LOSS_VALUE,
                                              None)

        loss = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE, None)

        assert_that(loss / self.VALID_SCALAR, equal_to(left_div_expected_result))
        assert_that(self.VALID_SCALAR / loss, equal_to(right__div_expected_result))

    def test_should_compute_the_mean(self):
        expected_result = ApexLoss(self.VALID_LOSS_ID, torch.tensor([2.0]), None)

        loss = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE, None)

        assert_that(loss.mean(), equal_to(expected_result))

    # noinspection PyTypeChecker
    def test_should_substract_a_scalar(self):
        left_sub_expected_result = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE - self.VALID_SCALAR, None)
        right_sub_expected_result = ApexLoss(self.VALID_LOSS_ID, self.VALID_SCALAR - self.VALID_LOSS_VALUE, None)

        loss = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE, None)

        assert_that(loss - self.VALID_SCALAR, equal_to(left_sub_expected_result))
        assert_that(self.VALID_SCALAR - loss, equal_to(right_sub_expected_result))


class ApexModuleTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def setUp(self) -> None:
        self._valid_optimizer_config = [{"type": "SGD", "params": {"lr": 0.01, "momentum": 0.5, "weight_decay": 0.0,
                                                                   "model_parameters": "bias"}},
                                        {"type": "SGD", "params": {"lr": 0.01, "momentum": 0.5, "weight_decay": 0.0,
                                                                   "model_parameters": "weight"}}]
        self._model = torch.nn.Conv2d(3, 32, (3, 3), (1, 1), bias=True)
        self._optimizer_bias = OptimizerFactory().create(self._valid_optimizer_config[0]["type"], self._model,
                                                         **self._valid_optimizer_config[0]["params"])
        self._optimizer_weights = OptimizerFactory().create(self._valid_optimizer_config[1]["type"], self._model,
                                                            **self._valid_optimizer_config[1]["params"])

        self._optimizers = {"weight": self._optimizer_weights, "bias": self._optimizer_bias}
        self._run_config = RunConfiguration()

    def test_should_initialize_optimizer_and_model(self):
        module = ApexModule(self._model, self._optimizers, 0, use_amp=True)

        assert_that(module.model, is_not(None))
        assert_that(module.optimizers, is_not(None))
        assert_that(module.optimizers, instance_of(dict))
