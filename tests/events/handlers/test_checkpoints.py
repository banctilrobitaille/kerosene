import os
import shutil
import unittest
from unittest.mock import PropertyMock

import mock
import mockito
import torch
from hamcrest import *
from ignite.metrics import Accuracy
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from kerosene.events import Event, MonitorMode, Monitor
from kerosene.events.handlers.checkpoints_2 import Checkpoint
from kerosene.metrics.gauges import AverageGauge
from kerosene.metrics.metrics import MetricType
from kerosene.training.trainers import SimpleTrainer, ModelTrainer


class ModelCheckpointIfBetterTest(unittest.TestCase):
    MODEL_NAME = "test_model"
    TRAINER_NAME = "test_trainer"
    SAVE_PATH = "tests/output/"

    def setUp(self) -> None:
        self._model_mock = mockito.mock(nn.Module)
        self._criterion_mock = mockito.mock(nn.CrossEntropyLoss)
        self._optimizer_mock = mockito.mock(Optimizer)
        self._scheduler_mock = mockito.mock(lr_scheduler)
        self._metric_computer_mock = mockito.mock(Accuracy)

        self._model_trainer = ModelTrainer(self.MODEL_NAME, self._model_mock, self._criterion_mock,
                                           self._optimizer_mock, self._scheduler_mock, self._metric_computer_mock)

        self._trainer_mock = mockito.mock(SimpleTrainer)

        self._valid_loss = AverageGauge()

    def tearDown(self) -> None:
        if os.path.exists(self.SAVE_PATH):
            shutil.rmtree(self.SAVE_PATH)

    @mock.patch("kerosene.training.trainers.ModelTrainer.optimizer_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.model_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.valid_loss", new_callable=PropertyMock)
    def test_should_not_save_model_with_higher_valid_loss(self, valid_loss_mock, model_state_mock,
                                                          optimizer_state_mock):
        self._handler_mock = mockito.spy(Checkpoint(self.SAVE_PATH, monitor=Monitor.VALID_LOSS))
        valid_loss_mock.return_value = torch.tensor([0.5])
        model_state_mock.return_value = dict
        optimizer_state_mock.return_value = dict
        self._trainer_mock.epoch = 5
        self._trainer_mock.model_trainers = [self._model_trainer]
        self._handler_mock(Event.ON_EPOCH_END, self._trainer_mock)
        valid_loss_mock.return_value = torch.tensor([0.6])
        assert_that(
            not os.path.exists(os.path.join(self.SAVE_PATH, self.MODEL_NAME, self.MODEL_NAME + "_optimizer.tar")))

    @mock.patch("kerosene.training.trainers.ModelTrainer.optimizer_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.model_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.valid_loss", new_callable=PropertyMock)
    def test_should_save_model_with_lower_valid_loss(self, valid_loss_mock, model_state_mock, optimizer_state_mock):
        self._handler_mock = mockito.spy(Checkpoint(self.SAVE_PATH, monitor=Monitor.VALID_LOSS))
        valid_loss_mock.return_value = torch.tensor([0.5])
        model_state_mock.return_value = dict
        optimizer_state_mock.return_value = dict
        self._trainer_mock.epoch = 5
        self._trainer_mock.model_trainers = [self._model_trainer]
        self._handler_mock(Event.ON_EPOCH_END, self._trainer_mock)
        valid_loss_mock.return_value = torch.tensor([0.4])
        self._handler_mock(Event.ON_EPOCH_END, self._trainer_mock)
        assert_that(os.path.exists(os.path.join(self.SAVE_PATH, self.MODEL_NAME, self.MODEL_NAME + ".tar")))

    @mock.patch("kerosene.training.trainers.ModelTrainer.optimizer_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.model_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.valid_loss", new_callable=PropertyMock)
    def test_should_save_optimizer_with_lower_valid_loss(self, valid_loss_mock, model_state_mock, optimizer_state_mock):
        self._handler_mock = mockito.spy(Checkpoint(self.SAVE_PATH, monitor=Monitor.VALID_LOSS))
        valid_loss_mock.return_value = torch.tensor([0.5])
        model_state_mock.return_value = dict
        optimizer_state_mock.return_value = dict
        self._trainer_mock.epoch = 5
        self._trainer_mock.model_trainers = [self._model_trainer]
        self._handler_mock(Event.ON_EPOCH_END, self._trainer_mock)
        valid_loss_mock.return_value = torch.tensor([0.4])
        self._handler_mock(Event.ON_EPOCH_END, self._trainer_mock)
        assert_that(
            os.path.exists(os.path.join(self.SAVE_PATH, self.MODEL_NAME, self.MODEL_NAME + "_optimizer.tar")))

    @mock.patch("kerosene.training.trainers.ModelTrainer.optimizer_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.model_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.valid_metrics", new_callable=PropertyMock)
    def test_should_save_optimizer_with_higher_valid_metric(self, valid_metrics_mock, model_state_mock, optimizer_state_mock):
        self._handler_mock = mockito.spy(
            Checkpoint(self.SAVE_PATH, monitor=Monitor.VALID_METRICS, metric_name=MetricType.Accuracy,
                       mode=MonitorMode.MAX))
        valid_metrics_mock.return_value = torch.tensor([0.5])
        model_state_mock.return_value = dict
        optimizer_state_mock.return_value = dict
        self._trainer_mock.epoch = 5
        self._trainer_mock.model_trainers = [self._model_trainer]
        self._handler_mock(Event.ON_EPOCH_END, self._trainer_mock)
        valid_metrics_mock.return_value = torch.tensor([0.8])
        self._handler_mock(Event.ON_EPOCH_END, self._trainer_mock)
        assert_that(
            os.path.exists(os.path.join(self.SAVE_PATH, self.MODEL_NAME, self.MODEL_NAME + "_optimizer.tar")))

    @mock.patch("kerosene.training.trainers.ModelTrainer.optimizer_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.model_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.valid_metrics", new_callable=PropertyMock)
    def test_should_not_save_optimizer_with_lower_valid_metric(self, valid_metrics_mock, model_state_mock,
                                                          optimizer_state_mock):
        self._handler_mock = mockito.spy(
            Checkpoint(self.SAVE_PATH, monitor=Monitor.VALID_METRICS, metric_name=MetricType.Accuracy,
                       mode=MonitorMode.MAX))
        valid_metrics_mock.return_value = torch.tensor([0.8])
        model_state_mock.return_value = dict
        optimizer_state_mock.return_value = dict
        self._trainer_mock.epoch = 5
        self._trainer_mock.model_trainers = [self._model_trainer]
        self._handler_mock(Event.ON_EPOCH_END, self._trainer_mock)
        valid_metrics_mock.return_value = torch.tensor([0.5])
        self._handler_mock(Event.ON_EPOCH_END, self._trainer_mock)
        assert_that(
            os.path.exists(os.path.join(self.SAVE_PATH, self.MODEL_NAME, self.MODEL_NAME + "_optimizer.tar")))