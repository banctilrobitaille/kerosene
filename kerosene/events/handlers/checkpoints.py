# -*- coding: utf-8 -*-
# Copyright 2019 Kerosene Authors. All Rights Reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

import torch

from kerosene.events import Monitor, MonitorMode, Event
from kerosene.events.handlers.base_handler import EventHandler
from kerosene.training.trainers import Trainer, ModelTrainer


class ModelCheckpoint(EventHandler):
    CHECKPOINT_EXT = ".tar"
    SUPPORTED_EVENTS = [Event.ON_EPOCH_END]

    def __init__(self, path, model_name=None, every=1):
        super(ModelCheckpoint, self).__init__(every)
        self._path = path
        self._model_name = model_name

    def __call__(self, event: Event, trainer: Trainer):
        assert event in self.SUPPORTED_EVENTS, "Unsupported event provided. Only {} are permitted.".format(
            self.SUPPORTED_EVENTS)
        model_trainer_states = list(filter(self._filter_by_name, trainer.model_trainers))

        for model_trainer_state in model_trainer_states:
            self._save_model(model_trainer_state.name, model_trainer_state.model_state)
            self._save_optimizer(model_trainer_state.name, model_trainer_state.optimizer_state)

    def _filter_by_name(self, model_trainer: ModelTrainer):
        return True if self._model_name is None else model_trainer.name == self._model_name

    def _save_model(self, model_name, model_state):
        torch.save(model_state, os.path.join(self._path, model_name, model_name + self.CHECKPOINT_EXT))

    def _save_optimizer(self, model_name, model_state):
        torch.save(model_state,
                   os.path.join(self._path, model_name, "{}_optimizer{}".format(model_name, self.CHECKPOINT_EXT)))


class ModelCheckpointIfBetter(EventHandler):
    CHECKPOINT_EXT = ".tar"
    SUPPORTED_EVENTS = [Event.ON_EPOCH_END]

    def __init__(self, path, monitor: Monitor = Monitor.VALID_LOSS, mode: MonitorMode = MonitorMode.MIN,
                 model_name=None, every=1):
        super(ModelCheckpointIfBetter, self).__init__(every)
        self._path = path
        self._monitor = monitor
        self._mode = mode
        self._model_name = model_name

        self._monitor_values = {}

    def __call__(self, event: Event, trainer: Trainer):
        assert event in self.SUPPORTED_EVENTS, "Unsupported event provided. Only {} are permitted.".format(
            self.SUPPORTED_EVENTS)

        model_trainer_states = list(filter(self._filter_by_name, trainer.model_trainers))

        for model_trainer_state in model_trainer_states:
            if model_trainer_state.name in self._monitor_values.keys() and self._should_save(model_trainer_state):
                self._monitor_values[model_trainer_state.name] = self._get_monitor_value(model_trainer_state)
                self._save_model(model_trainer_state.name, model_trainer_state.model_state)
                self._save_optimizer(model_trainer_state.name, model_trainer_state.optimizer_state)
            else:
                self._save_model(model_trainer_state.name, model_trainer_state.model_state)
                self._save_optimizer(model_trainer_state.name, model_trainer_state.optimizer_state)

    def _filter_by_name(self, model_trainer: ModelTrainer):
        return True if self._model_name is None else model_trainer.name == self._model_name

    def _save_model(self, model_name, model_state):
        torch.save(model_state, os.path.join(self._path, model_name, model_name + self.CHECKPOINT_EXT))

    def _save_optimizer(self, model_name, model_state):
        torch.save(model_state,
                   os.path.join(self._path, model_name, "{}_optimizer{}".format(model_name, self.CHECKPOINT_EXT)))

    def _should_save(self, model_trainer: ModelTrainer):
        if self._mode is MonitorMode.MIN:
            return self._get_monitor_value(model_trainer) < self._monitor_values[model_trainer.name]
        else:
            return self._get_monitor_value(model_trainer) > self._monitor_values[model_trainer.name]

    def _get_monitor_value(self, model_trainer: ModelTrainer):
        if self._monitor is Monitor.TRAINING_LOSS:
            value = model_trainer.train_loss
        elif self._monitor is Monitor.TRAINING_METRIC:
            value = model_trainer.train_metric
        elif self._monitor is Monitor.VALID_LOSS:
            value = model_trainer.valid_loss
        elif self._monitor is Monitor.VALID_METRIC:
            value = model_trainer.valid_metric
        else:
            raise NotImplementedError("The provided monitor value is not supported !")

        return value
