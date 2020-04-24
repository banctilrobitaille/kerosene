import logging

from kerosene.events import MonitorMode, TemporalEvent
from kerosene.events.handlers.base_monitor_watcher import MonitorWatcher, MonitorPatienceExceeded
from kerosene.training.trainers import Trainer


class EarlyStopping(MonitorWatcher):
    LOGGER = logging.getLogger("EarlyStopping")

    def __init__(self, monitor_fn, mode: MonitorMode, min_delta=0.01, patience=3):
        super(EarlyStopping, self).__init__(monitor_fn, mode, min_delta, patience)

    def __call__(self, temporal_event: TemporalEvent, monitors, trainer: Trainer):
        for model_trainer in trainer.model_trainers:
            try:
                value = self._monitor_fn(model_trainer)
                self.watch(model_trainer.name, value)
            except MonitorPatienceExceeded as e:
                model_trainer.finalize()
