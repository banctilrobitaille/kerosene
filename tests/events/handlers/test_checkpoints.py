import unittest
import os
import shutil
import torch
import mock
from ignite.metrics import Accuracy
import mockito
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from hamcrest import *
from kerosene.events import Event
from kerosene.events.handlers.checkpoints import ModelCheckpointIfBetter
from kerosene.metrics.gauges import AverageGauge
from kerosene.training.trainers import SimpleTrainer, ModelTrainer
from unittest.mock import PropertyMock


class ModelCheckpointIfBetterTest(unittest.TestCase):
    MODEL_NAME = "test_model"
    TRAINER_NAME = "test_trainer"
    SAVE_PATH = "tests/output/test_model_save/"

    def setUp(self) -> None:
        self._model_mock = mockito.mock(nn.Module)
        self._criterion_mock = mockito.mock(nn.CrossEntropyLoss)
        self._optimizer_mock = mockito.mock(Optimizer)
        self._scheduler_mock = mockito.mock(lr_scheduler)
        self._metric_computer_mock = mockito.mock(Accuracy)

        self._model_trainer = ModelTrainer(self.MODEL_NAME, self._model_mock, self._criterion_mock,
                                           self._optimizer_mock, self._scheduler_mock, self._metric_computer_mock)

        self._trainer_mock = mockito.mock(SimpleTrainer)
        self._handler_mock = mockito.spy(ModelCheckpointIfBetter(self.SAVE_PATH))
        self._valid_loss = AverageGauge()

    def tearDown(self) -> None:
        if os.path.exists(self.SAVE_PATH):
            shutil.rmtree(self.SAVE_PATH)

    @mock.patch("kerosene.training.trainers.ModelTrainer.model_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.optimizer_state", new_callable=PropertyMock)
    def test_should_save_model_with_lower_valid_loss(self, model_state_mock, optimizer_state_mock):
        with mock.patch("kerosene.training.trainers.ModelTrainer.valid_loss",
                        new_callable=PropertyMock) as valid_loss_mock:
            valid_loss_mock.return_value = torch.tensor([0.5])
            model_state_mock.return_value = dict
            optimizer_state_mock.return_value = dict
            self._trainer_mock.model_trainers = [self._model_trainer]
            self._handler_mock(Event.ON_EPOCH_END, self._trainer_mock)
            valid_loss_mock.return_value = torch.tensor([0.4])
            assert_that(self._handler_mock._should_save(self._model_trainer), is_(True))

    @mock.patch("kerosene.training.trainers.ModelTrainer.model_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.optimizer_state", new_callable=PropertyMock)
    def test_should_not_save_model_with_higher_valid_loss(self, model_state_mock, optimizer_state_mock):
        with mock.patch("kerosene.training.trainers.ModelTrainer.valid_loss",
                        new_callable=PropertyMock) as valid_loss_mock:
            valid_loss_mock.return_value = torch.tensor([0.5])
            model_state_mock.return_value = dict
            optimizer_state_mock.return_value = dict
            self._trainer_mock.model_trainers = [self._model_trainer]
            self._handler_mock(Event.ON_EPOCH_END, self._trainer_mock)
            valid_loss_mock.return_value = torch.tensor([0.6])
            assert_that(self._handler_mock._should_save(self._model_trainer), is_(False))

    @mock.patch("kerosene.training.trainers.ModelTrainer.model_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.optimizer_state", new_callable=PropertyMock)
    def test_should_create_model_tar_file(self, model_state_mock, optimizer_state_mock):
        with mock.patch("kerosene.training.trainers.ModelTrainer.valid_loss",
                        new_callable=PropertyMock) as valid_loss_mock:
            valid_loss_mock.return_value = torch.tensor([0.5])
            model_state_mock.return_value = dict
            optimizer_state_mock.return_value = dict
            self._trainer_mock.model_trainers = [self._model_trainer]
            self._handler_mock(Event.ON_EPOCH_END, self._trainer_mock)
            valid_loss_mock.return_value = torch.tensor([0.4])
            assert_that(os.path.exists(os.path.join(self.SAVE_PATH, self.MODEL_NAME, self.MODEL_NAME + ".tar")))

    @mock.patch("kerosene.training.trainers.ModelTrainer.model_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.optimizer_state", new_callable=PropertyMock)
    def test_should_create_optimizer_tar_file(self, model_state_mock, optimizer_state_mock):
        with mock.patch("kerosene.training.trainers.ModelTrainer.valid_loss",
                        new_callable=PropertyMock) as valid_loss_mock:
            valid_loss_mock.return_value = torch.tensor([0.5])
            model_state_mock.return_value = dict
            optimizer_state_mock.return_value = dict
            self._trainer_mock.model_trainers = [self._model_trainer]
            self._handler_mock(Event.ON_EPOCH_END, self._trainer_mock)
            valid_loss_mock.return_value = torch.tensor([0.4])
            assert_that(os.path.exists(os.path.join(self.SAVE_PATH, self.MODEL_NAME, self.MODEL_NAME + "_optimizer.tar")))
