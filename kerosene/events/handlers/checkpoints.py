import logging
import os
from typing import Callable, List, Union

import torch

from kerosene.events import MonitorMode, TemporalEvent, Phase, Monitor
from kerosene.events.handlers.base_monitor_watcher import MonitorWatcher, MonitorPatienceExceeded
from kerosene.training.events import Event
from kerosene.training.trainers import Trainer
from kerosene.utils.constants import CHECKPOINT_EXT
from kerosene.utils.files import should_create_dir


class Checkpoint(MonitorWatcher):
    LOGGER = logging.getLogger("Checkpoint")
    SUPPORTED_EVENTS = [Event.ON_EPOCH_END]

    def __init__(self, path: str, model_name: Union[str, None], monitor_names: Union[List[str], str], delta: float,
                 mode: MonitorMode, reduction: Callable = torch.mean):
        super(Checkpoint, self).__init__(mode, delta, patience=0, supported_events=self.SUPPORTED_EVENTS)
        self._path = path
        self._model_name = model_name
        self._reduction = reduction
        self._monitors_names = monitor_names if isinstance(monitor_names, list) else [monitor_names]

    def __call__(self, event: TemporalEvent, monitors: dict, trainer: Trainer):
        if self.should_handle(event):
            try:
                model_monitors = {**monitors[self._model_name][Phase.VALIDATION][Monitor.METRICS],
                                  **monitors[self._model_name][Phase.VALIDATION][Monitor.LOSS]}
                values = torch.cat([model_monitors[monitor_name] for monitor_name in self._monitors_names])

                self.watch(self._model_name, self._reduction(values))
            except MonitorPatienceExceeded as e:
                self._save(trainer.epoch, self._model_name, trainer.model_trainers[self._model_name].model_state,
                           trainer.model_trainers[self._model_name].optimizer_state)
            except KeyError as e:
                self.LOGGER.warning("Invalid model or monitor name: {}".format(e))

    def _save(self, epoch_num, model_name, model_state, optimizer_states):
        if should_create_dir(self._path, model_name):
            os.makedirs(os.path.join(self._path, model_name))
        torch.save({"epoch_num": epoch_num,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer_states},
                   os.path.join(self._path, model_name, model_name + CHECKPOINT_EXT))
