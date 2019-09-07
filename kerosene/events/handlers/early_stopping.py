from typing import Union, Dict

from kerosene.events import BaseVariable
from kerosene.events.handlers.base_handler import EventHandler
from kerosene.training import Status
from kerosene.training.state import TrainerState, ModelTrainerState


class EarlyStopping(EventHandler):
    def __init__(self, monitor: Union[BaseVariable, Dict[str, BaseVariable]], mode, min_delta, patience):
        self._monitor = monitor
        self._mode = mode
        self._min_delta = min_delta
        self._patience = patience

        self._monitor_values = {}

    def __call__(self, state: TrainerState):
        for model_trainer_state in state.model_trainer_states:
            if self._should_stop(model_trainer_state):
                model_trainer_state.status = Status.FINALIZE

    def _should_stop(self, model_trainer_state: ModelTrainerState):
        pass
