import unittest

import torch
from hamcrest import *
from ignite.metrics import Accuracy
from mockito import mock, verify, spy
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from kerosene.training.trainers import ModelTrainer


# noinspection PyUnresolvedReferences
class ModelTrainerTest(unittest.TestCase):
    MODEL_NAME = "Harry Potter"
    DELTA = 0.0001
    MODEL_PREDICTION_CLASS_0 = torch.tensor([[1.0, 0.0]])
    MODEL_PREDICTION_CLASS_1 = torch.tensor([[0.0, 1.0]])
    MINIMUM_BINARY_CROSS_ENTROPY_LOSS = torch.tensor(0.3133)
    TARGET_CLASS_0 = torch.tensor([0])
    TARGET_CLASS_1 = torch.tensor([1])

    def setUp(self) -> None:
        self._model_mock = mock(nn.Module)
        self._criterion_mock = spy(nn.CrossEntropyLoss())
        self._optimizer_mock = mock(Optimizer)
        self._scheduler_mock = mock(lr_scheduler)
        self._metric_computer_mock = spy(Accuracy())

        self._model_trainer = ModelTrainer(self.MODEL_NAME, self._model_mock, self._criterion_mock,
                                           self._optimizer_mock, self._scheduler_mock, self._metric_computer_mock)

    def tearDown(self) -> None:
        super().tearDown()

    def test_should_compute_train_metric_and_update_state(self):
        self._model_trainer.compute_train_metric(self.MODEL_PREDICTION_CLASS_0, self.TARGET_CLASS_0)
        assert_that(self._model_trainer.state.train_metric, equal_to(torch.tensor([1.0])))
        assert_that(self._model_trainer.state.step_train_metric, equal_to(torch.tensor([1.0])))
        verify(self._metric_computer_mock).update((self.MODEL_PREDICTION_CLASS_0, self.TARGET_CLASS_0))

        self._model_trainer.compute_train_metric(self.MODEL_PREDICTION_CLASS_1, self.TARGET_CLASS_0)
        assert_that(self._model_trainer.state.train_metric, equal_to(torch.tensor([0.5])))
        assert_that(self._model_trainer.state.step_train_metric, equal_to(torch.tensor([0.0])))

        verify(self._metric_computer_mock, times=2).compute()

        assert_that(self._model_trainer.state.valid_metric, equal_to(torch.tensor([0.0])))
        assert_that(self._model_trainer.state.step_valid_metric, equal_to(torch.tensor([0.0])))
        assert_that(self._model_trainer.state.step_train_loss, equal_to(torch.tensor([0.0])))
        assert_that(self._model_trainer.state.step_valid_loss, equal_to(torch.tensor([0.0])))
        assert_that(self._model_trainer.state.train_loss, equal_to(torch.tensor([0.0])))
        assert_that(self._model_trainer.state.valid_loss, equal_to(torch.tensor([0.0])))

    def test_should_compute_valid_metric_and_update_state(self):
        self._model_trainer.compute_valid_metric(self.MODEL_PREDICTION_CLASS_0, self.TARGET_CLASS_0)
        assert_that(self._model_trainer.state.valid_metric, equal_to(torch.tensor([1.0])))
        assert_that(self._model_trainer.state.step_valid_metric, equal_to(torch.tensor([1.0])))
        verify(self._metric_computer_mock).update((self.MODEL_PREDICTION_CLASS_0, self.TARGET_CLASS_0))

        self._model_trainer.compute_valid_metric(self.MODEL_PREDICTION_CLASS_1, self.TARGET_CLASS_0)
        assert_that(self._model_trainer.state.valid_metric, equal_to(torch.tensor([0.5])))
        assert_that(self._model_trainer.state.step_valid_metric, equal_to(torch.tensor([0.0])))

        verify(self._metric_computer_mock, times=2).compute()

        assert_that(self._model_trainer.state.train_metric, equal_to(torch.tensor([0.0])))
        assert_that(self._model_trainer.state.step_train_metric, equal_to(torch.tensor([0.0])))
        assert_that(self._model_trainer.state.step_train_loss, equal_to(torch.tensor([0.0])))
        assert_that(self._model_trainer.state.step_valid_loss, equal_to(torch.tensor([0.0])))
        assert_that(self._model_trainer.state.train_loss, equal_to(torch.tensor([0.0])))
        assert_that(self._model_trainer.state.valid_loss, equal_to(torch.tensor([0.0])))

    def test_should_compute_train_loss_and_update_state(self):
        loss = self._model_trainer.compute_train_loss(self.MODEL_PREDICTION_CLASS_0, self.TARGET_CLASS_0)
        assert_that(loss._loss, close_to(self.MINIMUM_BINARY_CROSS_ENTROPY_LOSS, self.DELTA))

    def test_should_compute_valid_loss_and_update_state(self):
        pass
