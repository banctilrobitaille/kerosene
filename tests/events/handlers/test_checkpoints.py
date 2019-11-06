import unittest

import torch
import mock
from ignite.metrics import Accuracy
from mockito import spy, verify, when
from mockito import mock as mockito_mock
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from kerosene.events import Event
from kerosene.events.handlers.checkpoints import ModelCheckpointIfBetter
from kerosene.training.trainers import SimpleTrainer, ModelTrainer


class ModelCheckpointIfBetterTest(unittest.TestCase):
    MODEL_NAME = "test_model"
    TRAINER_NAME = "test_trainer"

    def setUp(self) -> None:
        self._model_mock = mockito_mock(nn.Module)
        self._criterion_mock = mockito_mock(nn.CrossEntropyLoss())
        self._optimizer_mock = mockito_mock(Optimizer)
        self._scheduler_mock = mockito_mock(lr_scheduler)
        self._metric_computer_mock = mockito_mock(Accuracy())

        self._model_trainer = ModelTrainer(self.MODEL_NAME, self._model_mock, self._criterion_mock,
                                           self._optimizer_mock, self._scheduler_mock, self._metric_computer_mock)

        self._trainer_mock = mockito_mock(SimpleTrainer)
        self._handler_mock = spy(ModelCheckpointIfBetter(path="."))

    def test_should_save_model_with_lower_valid_loss(self):
        model_trainer_prop = mock.PropertyMock(return_value=[self._model_trainer])
        type(self._trainer_mock).model_trainers = model_trainer_prop
        model_state_prop = mock.PropertyMock(return_value={})
        type(self._model_trainer).model_state = model_state_prop
        optim_state_prop = mock.PropertyMock(return_value={})
        type(self._model_trainer).optimizer_state = optim_state_prop
        self._model_trainer._valid_loss = torch.tensor([0.5])
        self._handler_mock(Event.ON_EPOCH_END, self._trainer_mock)
        self._model_trainer._valid_loss = torch.tensor([0.4])
        self._handler_mock(Event.ON_EPOCH_END, self._trainer_mock)
        verify(self._handler_mock, times=2)._should_save(self.MODEL_NAME, model_state_prop)

