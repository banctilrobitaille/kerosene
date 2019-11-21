import logging
import os

import torch
from ignite.metrics import Metric

from kerosene.events import Event, Monitor, MonitorMode
from kerosene.events.exceptions import UnsupportedEventException
from kerosene.events.handlers.base_monitor_watcher import MonitorWatcher, MonitorPatienceExceeded
from kerosene.training.trainers import Trainer
from kerosene.utils.constants import CHECKPOINT_EXT
from kerosene.utils.files import should_create_model_dir, create_model_dir


class Checkpoint(MonitorWatcher):
    LOGGER = logging.getLogger("Checkpoint")
    CHECKPOINT_EXT = ".tar"
    SUPPORTED_EVENTS = [Event.ON_EPOCH_END]

    def __init__(self, path, monitor: Monitor, mode: MonitorMode = MonitorMode.MIN):
        super(Checkpoint, self).__init__(monitor, mode, patience=0)
        self._path = path

    def __call__(self, event: Event, trainer: Trainer):
        if event not in self.SUPPORTED_EVENTS:
            raise UnsupportedEventException(event, self.SUPPORTED_EVENTS)

        for model_trainer in trainer.model_trainers:
            try:
                value = getattr(model_trainer, str(self._monitor))
                self.watch(model_trainer.name, value)
            except MonitorPatienceExceeded as e:
                self._save_model(model_trainer.name, model_trainer.model_state, trainer.epoch)
                self._save_optimizer(model_trainer.name, model_trainer.optimizer_state)

    def _save_model(self, model_name, model_state, epoch_num):
        if should_create_model_dir(self._path, model_name):
            create_model_dir(self._path, model_name)
        torch.save({"model_state_dict": model_state, "epoch_num": epoch_num},
                   os.path.join(self._path, model_name, model_name + CHECKPOINT_EXT))

    def _save_optimizer(self, model_name, optimizer_state):
        if should_create_model_dir(self._path, model_name):
            create_model_dir(self._path, model_name)
        torch.save({"optimizer_state_dict": optimizer_state},
                   os.path.join(self._path, model_name, "{}_optimizer{}".format(model_name, CHECKPOINT_EXT)))
