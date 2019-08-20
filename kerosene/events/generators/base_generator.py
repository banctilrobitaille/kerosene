from abc import ABC, abstractmethod
from typing import Callable

from kerosene.events import Event


class EventGenerator(ABC):
    def __init__(self):
        self._event_handlers = {}

    @property
    @abstractmethod
    def state(self):
        raise NotImplementedError()

    @abstractmethod
    def with_event_handler(self, handler, event: Event, preprocessor: Callable):
        raise NotImplementedError()

    def fire(self, event: Event):
        if event in self._event_handlers.keys():
            for handler in self._event_handlers[event]:
                handler(event, self.state)
