import logging

from kerosene.events.handlers.base_monitor_watcher import MonitorWatcher, MonitorPatienceExceeded
from kerosene.training.trainers import Trainer


class EarlyStopping(MonitorWatcher):
    LOGGER = logging.getLogger("EarlyStopping")

    def __call__(self, trainer: Trainer):
        for model_trainer in trainer.model_trainers:
            try:
                value = getattr(model_trainer, str(self._monitor))
                self.watch(model_trainer.name, value)
            except MonitorPatienceExceeded as e:
                model_trainer._finalize()
