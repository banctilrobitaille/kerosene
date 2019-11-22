import unittest

from hamcrest import *
from ignite.metrics import Accuracy
from mockito import mock, verify, spy
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from kerosene.training.trainers import ModelTrainer
from tests.constants import *


# noinspection PyUnresolvedReferences
class ModelTrainerTest(unittest.TestCase):
    MODEL_NAME = "Harry Potter"

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
        metric = self._model_trainer.compute_metric(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0)
        self._model_trainer.update_train_metric(metric)
        assert_that(self._model_trainer.train_metric, equal_to(ONE))
        assert_that(self._model_trainer.step_train_metric, equal_to(ONE))
        verify(self._metric_computer_mock).update((MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0))

        metric = self._model_trainer.compute_metric(MODEL_PREDICTION_CLASS_1, TARGET_CLASS_0)
        self._model_trainer.update_train_metric(metric)
        assert_that(self._model_trainer.train_metric, equal_to(ONE / 2))
        assert_that(self._model_trainer.step_train_metric, equal_to(ZERO))

        verify(self._metric_computer_mock, times=2).compute()

        assert_that(self._model_trainer.valid_metric, equal_to(ZERO))
        assert_that(self._model_trainer.test_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_test_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_test_loss, equal_to(ZERO))
        assert_that(self._model_trainer.train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.test_loss, equal_to(ZERO))

    def test_should_compute_valid_metric_and_update_state(self):
        metric = self._model_trainer.compute_metric(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0)
        self._model_trainer.update_valid_metric(metric)
        assert_that(self._model_trainer.valid_metric, equal_to(ONE))
        assert_that(self._model_trainer.step_valid_metric, equal_to(ONE))
        verify(self._metric_computer_mock).update((MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0))

        metric = self._model_trainer.compute_metric(MODEL_PREDICTION_CLASS_1, TARGET_CLASS_0)
        self._model_trainer.update_valid_metric(metric)
        assert_that(self._model_trainer.valid_metric, equal_to(ONE / 2))
        assert_that(self._model_trainer.step_valid_metric, equal_to(ZERO))

        verify(self._metric_computer_mock, times=2).compute()

        assert_that(self._model_trainer.train_metric, equal_to(ZERO))
        assert_that(self._model_trainer.test_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_train_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_test_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_test_loss, equal_to(ZERO))
        assert_that(self._model_trainer.train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.test_loss, equal_to(ZERO))

    def test_should_compute_test_metric_and_update_state(self):
        metric = self._model_trainer.compute_metric(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0)
        self._model_trainer.update_test_metric(metric)
        assert_that(self._model_trainer.test_metric, equal_to(ONE))
        assert_that(self._model_trainer.step_test_metric, equal_to(ONE))
        verify(self._metric_computer_mock).update((MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0))

        metric = self._model_trainer.compute_metric(MODEL_PREDICTION_CLASS_1, TARGET_CLASS_0)
        self._model_trainer.update_test_metric(metric)
        assert_that(self._model_trainer.test_metric, equal_to(ONE / 2))
        assert_that(self._model_trainer.step_test_metric, equal_to(ZERO))

        verify(self._metric_computer_mock, times=2).compute()

        assert_that(self._model_trainer.train_metric, equal_to(ZERO))
        assert_that(self._model_trainer.valid_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_train_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_test_loss, equal_to(ZERO))
        assert_that(self._model_trainer.train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.test_loss, equal_to(ZERO))

    def test_should_compute_train_loss_and_update_state(self):
        loss = self._model_trainer.compute_and_update_train_loss(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0)

        assert_that(loss._loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_train_loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.train_loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        loss = self._model_trainer.compute_and_update_train_loss(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_1)
        assert_that(loss._loss, close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_train_loss, close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.train_loss, close_to(AVERAGED_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        assert_that(self._model_trainer.train_metric, equal_to(ZERO))
        assert_that(self._model_trainer.valid_metric, equal_to(ZERO))
        assert_that(self._model_trainer.test_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_train_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_test_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_test_loss, equal_to(ZERO))
        assert_that(self._model_trainer.valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.test_loss, equal_to(ZERO))

    def test_should_compute_valid_loss_and_update_state(self):
        loss = self._model_trainer.compute_and_update_valid_loss(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0)

        assert_that(loss._loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_valid_loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.valid_loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        loss = self._model_trainer.compute_and_update_valid_loss(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_1)
        assert_that(loss._loss, close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_valid_loss, close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.valid_loss, close_to(AVERAGED_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        assert_that(self._model_trainer.train_metric, equal_to(ZERO))
        assert_that(self._model_trainer.valid_metric, equal_to(ZERO))
        assert_that(self._model_trainer.test_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_train_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_test_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_test_loss, equal_to(ZERO))
        assert_that(self._model_trainer.train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.test_loss, equal_to(ZERO))

    def test_should_compute_test_loss_and_update_state(self):
        loss = self._model_trainer.compute_and_update_test_loss(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0)

        assert_that(loss._loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_test_loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.test_loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        loss = self._model_trainer.compute_and_update_test_loss(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_1)
        assert_that(loss._loss, close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_test_loss, close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.test_loss, close_to(AVERAGED_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        assert_that(self._model_trainer.train_metric, equal_to(ZERO))
        assert_that(self._model_trainer.valid_metric, equal_to(ZERO))
        assert_that(self._model_trainer.test_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_train_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_test_metric, equal_to(ZERO))
        assert_that(self._model_trainer.step_train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.valid_loss, equal_to(ZERO))
