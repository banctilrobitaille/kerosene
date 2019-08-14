from abc import ABC, abstractmethod
from typing import Callable


class EventHandler(ABC):

    @abstractmethod
    def __call__(self, *inputs):
        raise NotImplementedError()


class HandlerProcessor(object):
    def __init__(self, handler: EventHandler, preprocessor: Callable = (lambda x: x)):
        self._handler = handler
        self._preprocessor = preprocessor

    def __call__(self, inputs):
        processed_inputs = self._preprocessor(inputs)
        self._handler(*processed_inputs)
