import logging

from kerosene.events.handlers.base_monitor_inspector import MonitorInspector, MonitorFailedInspection
from kerosene.training.trainers import Trainer


class EarlyStopping(MonitorInspector):
    LOGGER = logging.getLogger("EarlyStopping")

    def __call__(self, trainer: Trainer):
        for model_trainer in trainer.model_trainers:
            try:
                value = getattr(model_trainer, str(self._monitor))
                self.inspect(model_trainer.name, value)
            except MonitorFailedInspection as e:
                model_trainer.finalize()
