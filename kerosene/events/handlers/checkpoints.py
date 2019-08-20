import os

import torch

from kerosene.events import Monitor, MonitorMode
from kerosene.events.handlers.base_handler import EventHandler
from kerosene.training.state import TrainerState, ModelTrainerState


class ModelCheckpoint(EventHandler):
    CHECKPOINT_EXT = ".tar"

    def __init__(self, path, model_name=None):
        self._path = path
        self._model_name = model_name

    def __call__(self, state: TrainerState):
        model_trainer_states = list(filter(self._filter_by_name, state.model_trainer_states))

        for model_trainer_state in model_trainer_states:
            self._save_model(model_trainer_state.name, model_trainer_state.model_state)
            self._save_optimizer(model_trainer_state.name, model_trainer_state.optimizer_state)

    def _filter_by_name(self, model_trainer_state: ModelTrainerState):
        return True if self._model_name is None else model_trainer_state.name == self._model_name

    def _save_model(self, model_name, model_state):
        torch.save(model_state, os.path.join(self._path, model_name, model_name + self.CHECKPOINT_EXT))

    def _save_optimizer(self, model_name, model_state):
        torch.save(model_state,
                   os.path.join(self._path, model_name, "{}_optimizer{}".format(model_name, self.CHECKPOINT_EXT)))


class ModelCheckpointIfBetter(EventHandler):
    CHECKPOINT_EXT = ".tar"

    def __init__(self, path, monitor: Monitor = Monitor.VALIDATION_LOSS, mode: MonitorMode = MonitorMode.MIN,
                 model_name=None):
        self._path = path
        self._monitor = monitor
        self._mode = mode
        self._model_name = model_name

        self._monitor_values = {}

    def __call__(self, state: TrainerState):
        model_trainer_states = list(filter(self._filter_by_name, state.model_trainer_states))

        for model_trainer_state in model_trainer_states:
            if model_trainer_state.name in self._monitor_values.keys() and self._should_save(model_trainer_state):
                self._monitor_values[model_trainer_state.name] = self._get_monitor_value(model_trainer_state)
                self._save_model(model_trainer_state.name, model_trainer_state.model_state)
                self._save_optimizer(model_trainer_state.name, model_trainer_state.optimizer_state)
            else:
                self._save_model(model_trainer_state.name, model_trainer_state.model_state)
                self._save_optimizer(model_trainer_state.name, model_trainer_state.optimizer_state)

    def _filter_by_name(self, model_trainer_state: ModelTrainerState):
        return True if self._model_name is None else model_trainer_state.name == self._model_name

    def _save_model(self, model_name, model_state):
        torch.save(model_state, os.path.join(self._path, model_name, model_name + self.CHECKPOINT_EXT))

    def _save_optimizer(self, model_name, model_state):
        torch.save(model_state,
                   os.path.join(self._path, model_name, "{}_optimizer{}".format(model_name, self.CHECKPOINT_EXT)))

    def _should_save(self, model_trainer_state: ModelTrainerState):
        if self._mode is MonitorMode.MIN:
            return self._get_monitor_value(model_trainer_state) < self._monitor_values[model_trainer_state.name]
        else:
            return self._get_monitor_value(model_trainer_state) > self._monitor_values[model_trainer_state.name]

    def _get_monitor_value(self, model_trainer_state: ModelTrainerState):
        if self._monitor is Monitor.TRAINING_LOSS:
            value = model_trainer_state.train_loss
        elif self._monitor is Monitor.TRAINING_METRIC:
            value = model_trainer_state.train_metric
        elif self._monitor is Monitor.VALIDATION_LOSS:
            value = model_trainer_state.valid_loss
        elif self._monitor is Monitor.VALIDATION_METRIC:
            value = model_trainer_state.valid_metric
        else:
            raise NotImplementedError("The provided monitor value is not supported !")

        return value
