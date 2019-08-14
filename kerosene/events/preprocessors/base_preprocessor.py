from abc import ABC, abstractmethod
from typing import Callable

from kerosene.events.handlers.base_handler import EventHandler
from kerosene.training.state import TrainerState


class EventPreprocessor(ABC):

    @abstractmethod
    def __call__(self, state: TrainerState):
        raise NotImplementedError()


class Identity(EventPreprocessor):

    def __call__(self, state: TrainerState):
        return state


class HandlerPreprocessor(EventPreprocessor):
    def __init__(self, handler: EventHandler, preprocessor: Callable = Identity()):
        self._handler = handler
        self._preprocessor = preprocessor

    def __call__(self, state: TrainerState):
        processed_state = self._preprocessor(state)
        self._handler(*processed_state if hasattr(processed_state, '__iter__') else [processed_state])
