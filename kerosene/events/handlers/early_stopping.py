from kerosene.events import Monitor
from kerosene.events.handlers.base_monitor_inspector import MonitorInspector, MonitorFailedInspection
from kerosene.training import Status
from kerosene.training.state import TrainerState


class EarlyStopping(MonitorInspector):

    def __call__(self, state: TrainerState):
        for model_trainer_state in state.model_trainer_states:
            try:
                if isinstance(self._monitor, Monitor):
                    value = getattr(model_trainer_state, str(self._monitor))
                else:
                    value = state.custom_variables[self._monitor]
                self.inspect(model_trainer_state.name, value)
            except MonitorFailedInspection as e:
                model_trainer_state.status = Status.FINALIZE
