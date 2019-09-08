from kerosene.events import Monitor
from kerosene.events.handlers.base_monitor_inspector import MonitorInspector, MonitorFailedInspection
from kerosene.training import Status
from kerosene.training.trainers import Trainer


class EarlyStopping(MonitorInspector):

    def __call__(self, state: Trainer):
        for model_trainer in state.model_trainers:
            try:
                if isinstance(self._monitor, Monitor):
                    value = getattr(model_trainer, str(self._monitor))
                else:
                    value = state.custom_variables[self._monitor]
                self.inspect(model_trainer.name, value)
            except MonitorFailedInspection as e:
                model_trainer.status = Status.FINALIZE
