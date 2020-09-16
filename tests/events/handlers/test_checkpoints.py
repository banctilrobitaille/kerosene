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

from kerosene.events import MonitorMode, Moment, Frequency, Phase, TemporalEvent, Monitor
from kerosene.events.handlers.checkpoints import Checkpoint
from kerosene.nn.utils.gradients import GradientClippingStrategy
from kerosene.training.events import Event
from kerosene.training.trainers import SimpleTrainer, ModelTrainer, ModelTrainerList


class ModelCheckpointIfBetterTest(unittest.TestCase):
    MODEL_NAME = "test_model"
    TRAINER_NAME = "test_trainer"
    SAVE_PATH = "tests/output/"

    def setUp(self) -> None:
        self._model_mock = mockito.mock(nn.Module)
        self._criterion_mock = {"CrossEntropyLoss": mockito.mock(nn.CrossEntropyLoss)}
        self._optimizer_mock = mockito.mock(Optimizer)
        self._scheduler_mock = mockito.mock(lr_scheduler)
        self._metric_computer_mock = {"Accuracy": mockito.mock(Accuracy)}
        self._gradient_clipping_strategy = mockito.mock(GradientClippingStrategy)

        self._model_trainer = ModelTrainer(self.MODEL_NAME, self._model_mock, self._criterion_mock,
                                           self._optimizer_mock, self._scheduler_mock, self._metric_computer_mock,
                                           self._gradient_clipping_strategy)

        self._trainer_mock = mockito.mock(SimpleTrainer)
        self._trainer_mock.epoch = 0
        self._trainer_mock.model_trainers = ModelTrainerList([self._model_trainer])

    def tearDown(self) -> None:
        if os.path.exists(self.SAVE_PATH):
            shutil.rmtree(self.SAVE_PATH)

    @mock.patch("kerosene.training.trainers.ModelTrainer.optimizer_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.model_state", new_callable=PropertyMock)
    def test_should_not_save_model_with_higher_valid_loss(self, model_state_mock, optimizer_states_mock):
        model_state_mock.return_value = dict()
        optimizer_states_mock.return_value = list(dict())
        moment = Moment(200, Frequency.EPOCH, Phase.VALIDATION)

        handler_mock = mockito.spy(Checkpoint(self.SAVE_PATH, self.MODEL_NAME, "MSELoss", 0.01, MonitorMode.MIN))

        monitors = {self.MODEL_NAME: {Phase.TRAINING: {Monitor.METRICS: {}, Monitor.LOSS: {}},
                                      Phase.VALIDATION: {Monitor.METRICS: {},
                                                         Monitor.LOSS: {"MSELoss": torch.tensor([0.5])}},
                                      Phase.TEST: {Monitor.METRICS: {}, Monitor.LOSS: {}}}}

        handler_mock(TemporalEvent(Event.ON_EPOCH_END, moment), monitors, self._trainer_mock)

        monitors = {self.MODEL_NAME: {Phase.TRAINING: {Monitor.METRICS: {}, Monitor.LOSS: {}},
                                      Phase.VALIDATION: {Monitor.METRICS: {},
                                                         Monitor.LOSS: {"MSELoss": torch.tensor([0.6])}},
                                      Phase.TEST: {Monitor.METRICS: {}, Monitor.LOSS: {}}}}

        handler_mock(TemporalEvent(Event.ON_EPOCH_END, moment), monitors, self._trainer_mock)

        assert_that(not os.path.exists(os.path.join(self.SAVE_PATH, self.MODEL_NAME, self.MODEL_NAME + ".tar")))

    @mock.patch("kerosene.training.trainers.ModelTrainer.optimizer_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.model_state", new_callable=PropertyMock)
    def test_should_not_save_model_with_higher_valid_losses(self, model_state_mock, optimizer_states_mock):
        model_state_mock.return_value = dict()
        optimizer_states_mock.return_value = list(dict())
        moment = Moment(200, Frequency.EPOCH, Phase.VALIDATION)

        handler_mock = mockito.spy(
            Checkpoint(self.SAVE_PATH, self.MODEL_NAME, ["MSELoss", "L1Loss"], 0.01, MonitorMode.MIN))

        monitors = {self.MODEL_NAME: {Phase.TRAINING: {Monitor.METRICS: {}, Monitor.LOSS: {}},
                                      Phase.VALIDATION: {Monitor.METRICS: {},
                                                         Monitor.LOSS: {"MSELoss": torch.tensor([0.5]),
                                                                        "L1Loss": torch.tensor([0.5])}},
                                      Phase.TEST: {Monitor.METRICS: {}, Monitor.LOSS: {}}}}

        handler_mock(TemporalEvent(Event.ON_EPOCH_END, moment), monitors, self._trainer_mock)

        monitors = {self.MODEL_NAME: {Phase.TRAINING: {Monitor.METRICS: {}, Monitor.LOSS: {}},
                                      Phase.VALIDATION: {Monitor.METRICS: {},
                                                         Monitor.LOSS: {"MSELoss": torch.tensor([0.5]),
                                                                        "L1Loss": torch.tensor([0.6])}},
                                      Phase.TEST: {Monitor.METRICS: {}, Monitor.LOSS: {}}}}

        handler_mock(TemporalEvent(Event.ON_EPOCH_END, moment), monitors, self._trainer_mock)

        assert_that(not os.path.exists(os.path.join(self.SAVE_PATH, self.MODEL_NAME, self.MODEL_NAME + ".tar")))

    @mock.patch("kerosene.training.trainers.ModelTrainer.optimizer_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.model_state", new_callable=PropertyMock)
    def test_should_save_model_with_lower_valid_loss(self, model_state_mock, optimizer_states_mock):
        model_state_mock.return_value = dict()
        optimizer_states_mock.return_value = list(dict())
        moment = Moment(200, Frequency.EPOCH, Phase.VALIDATION)

        handler_mock = mockito.spy(Checkpoint(self.SAVE_PATH, self.MODEL_NAME, "MSELoss", 0.01, MonitorMode.MIN))

        monitors = {self.MODEL_NAME: {Phase.TRAINING: {Monitor.METRICS: {}, Monitor.LOSS: {}},
                                      Phase.VALIDATION: {Monitor.METRICS: {},
                                                         Monitor.LOSS: {"MSELoss": torch.tensor([0.5])}},
                                      Phase.TEST: {Monitor.METRICS: {}, Monitor.LOSS: {}}}}

        handler_mock(TemporalEvent(Event.ON_EPOCH_END, moment), monitors, self._trainer_mock)

        monitors = {self.MODEL_NAME: {Phase.TRAINING: {Monitor.METRICS: {}, Monitor.LOSS: {}},
                                      Phase.VALIDATION: {Monitor.METRICS: {},
                                                         Monitor.LOSS: {"MSELoss": torch.tensor([0.3])}},
                                      Phase.TEST: {Monitor.METRICS: {}, Monitor.LOSS: {}}}}

        handler_mock(TemporalEvent(Event.ON_EPOCH_END, moment), monitors, self._trainer_mock)

        assert_that(os.path.exists(os.path.join(self.SAVE_PATH, self.MODEL_NAME, self.MODEL_NAME + ".tar")))

    @mock.patch("kerosene.training.trainers.ModelTrainer.optimizer_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.model_state", new_callable=PropertyMock)
    def test_should_save_model_with_higher_valid_metric(self, model_state_mock, optimizer_states_mock):
        model_state_mock.return_value = dict()
        optimizer_states_mock.return_value = list(dict())
        moment = Moment(200, Frequency.EPOCH, Phase.VALIDATION)

        handler_mock = mockito.spy(Checkpoint(self.SAVE_PATH, self.MODEL_NAME, "Accuracy", 0.01, MonitorMode.MAX))

        monitors = {self.MODEL_NAME: {Phase.TRAINING: {Monitor.METRICS: {}, Monitor.LOSS: {}},
                                      Phase.VALIDATION: {Monitor.METRICS: {"Accuracy": torch.tensor([0.5])},
                                                         Monitor.LOSS: {}},
                                      Phase.TEST: {Monitor.METRICS: {}, Monitor.LOSS: {}}}}

        handler_mock(TemporalEvent(Event.ON_EPOCH_END, moment), monitors, self._trainer_mock)

        monitors = {self.MODEL_NAME: {Phase.TRAINING: {Monitor.METRICS: {}, Monitor.LOSS: {}},
                                      Phase.VALIDATION: {Monitor.METRICS: {"Accuracy": torch.tensor([0.6])},
                                                         Monitor.LOSS: {}},
                                      Phase.TEST: {Monitor.METRICS: {}, Monitor.LOSS: {}}}}

        handler_mock(TemporalEvent(Event.ON_EPOCH_END, moment), monitors, self._trainer_mock)

        assert_that(os.path.exists(os.path.join(self.SAVE_PATH, self.MODEL_NAME, self.MODEL_NAME + ".tar")))

    @mock.patch("kerosene.training.trainers.ModelTrainer.optimizer_state", new_callable=PropertyMock)
    @mock.patch("kerosene.training.trainers.ModelTrainer.model_state", new_callable=PropertyMock)
    def test_should_not_save_model_with_lower_valid_metric(self, model_state_mock, optimizer_states_mock):
        model_state_mock.return_value = dict()
        optimizer_states_mock.return_value = list(dict())
        moment = Moment(200, Frequency.EPOCH, Phase.VALIDATION)

        handler_mock = mockito.spy(Checkpoint(self.SAVE_PATH, self.MODEL_NAME, "Accuracy", 0.01, MonitorMode.MAX))

        monitors = {self.MODEL_NAME: {Phase.TRAINING: {Monitor.METRICS: {}, Monitor.LOSS: {}},
                                      Phase.VALIDATION: {Monitor.METRICS: {"Accuracy": torch.tensor([0.5])},
                                                         Monitor.LOSS: {}},
                                      Phase.TEST: {Monitor.METRICS: {}, Monitor.LOSS: {}}}}

        handler_mock(TemporalEvent(Event.ON_EPOCH_END, moment), monitors, self._trainer_mock)

        monitors = {self.MODEL_NAME: {Phase.TRAINING: {Monitor.METRICS: {}, Monitor.LOSS: {}},
                                      Phase.VALIDATION: {Monitor.METRICS: {"Accuracy": torch.tensor([0.4])},
                                                         Monitor.LOSS: {}},
                                      Phase.TEST: {Monitor.METRICS: {}, Monitor.LOSS: {}}}}

        handler_mock(TemporalEvent(Event.ON_EPOCH_END, moment), monitors, self._trainer_mock)

        assert_that(not os.path.exists(os.path.join(self.SAVE_PATH, self.MODEL_NAME, self.MODEL_NAME + ".tar")))
