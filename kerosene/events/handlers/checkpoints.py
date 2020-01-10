import logging
import os
from typing import Callable

import torch

from kerosene.events import Event, MonitorMode, TemporalEvent
from kerosene.events.exceptions import UnsupportedEventException
from kerosene.events.handlers.base_monitor_watcher import MonitorWatcher, MonitorPatienceExceeded
from kerosene.training.trainers import Trainer
from kerosene.utils.constants import CHECKPOINT_EXT
from kerosene.utils.files import should_create_dir


class Checkpoint(MonitorWatcher):
    LOGGER = logging.getLogger("Checkpoint")
    SUPPORTED_EVENTS = [Event.ON_EPOCH_END]

    def __init__(self, path, monitor_fn: Callable, delta: float, mode: MonitorMode):
        super(Checkpoint, self).__init__(monitor_fn, mode, delta, patience=0)
        self._path = path

    def __call__(self, temporal_event: TemporalEvent, monitors: dict, trainer: Trainer):
        if temporal_event.event not in self.SUPPORTED_EVENTS:
            raise UnsupportedEventException(temporal_event.event, self.SUPPORTED_EVENTS)

        for model_trainer in trainer.model_trainers:
            try:
                value = self._monitor_fn(model_trainer)
                self.watch(model_trainer.name, value)
            except MonitorPatienceExceeded as e:
                self._save_model(model_trainer.name, model_trainer.model_state, trainer.epoch)
                self._save_optimizer(model_trainer.name, model_trainer.optimizer_state)

    def _save_model(self, model_name, model_state, epoch_num):
        if should_create_dir(self._path, model_name):
            os.makedirs(os.path.join(self._path, model_name))
        torch.save({"model_state_dict": model_state, "epoch_num": epoch_num},
                   os.path.join(self._path, model_name, model_name + CHECKPOINT_EXT))

    def _save_optimizer(self, model_name, optimizer_state):
        if should_create_dir(self._path, model_name):
            os.makedirs(os.path.join(self._path, model_name))
        torch.save({"optimizer_state_dict": optimizer_state},
                   os.path.join(self._path, model_name, "{}_optimizer{}".format(model_name, CHECKPOINT_EXT)))
