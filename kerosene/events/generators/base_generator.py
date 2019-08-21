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
            state = self.state

            for handler in self._event_handlers[event]:
                handler(event, state)

    @abstractmethod
    def _on_epoch_begin(self):
        raise NotImplementedError()

    @abstractmethod
    def _on_epoch_end(self):
        raise NotImplementedError()

    @abstractmethod
    def on_epoch_begin(self):
        pass

    @abstractmethod
    def on_epoch_end(self):
        pass
