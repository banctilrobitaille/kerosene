import logging
import os
from typing import Callable

import torch

from kerosene.events import Event, MonitorMode
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

    def __call__(self, event: Event, trainer: Trainer):
        if event not in self.SUPPORTED_EVENTS:
            raise UnsupportedEventException(event, self.SUPPORTED_EVENTS)

        for model_trainer in trainer.model_trainers:
            try:
                value = self._monitor_fn(model_trainer)
                self.watch(model_trainer.name, value)
            except MonitorPatienceExceeded as e:
                self._save(trainer.epoch, model_trainer.name, model_trainer.model_state, model_trainer.optimizer_states)

    def _save(self, epoch_num, model_name, model_state, optimizer_states):
        if should_create_dir(self._path, model_name):
            os.makedirs(os.path.join(self._path, model_name))
        torch.save({"epoch_num": epoch_num,
                    "model_state_dict": model_state,
                    "optimizer_state": optimizer_states},
                   os.path.join(self._path, model_name, model_name + CHECKPOINT_EXT))
