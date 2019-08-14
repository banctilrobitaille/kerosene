from abc import ABC, abstractmethod
from typing import Callable

from kerosene.events.preprocessors import Identity


class EventHandler(ABC):

    @abstractmethod
    def __call__(self, inputs):
        raise NotImplementedError()


class HandlerPreprocessor(EventHandler):
    def __init__(self, handler: EventHandler, preprocessor: Callable):
        self._handler = handler
        self._preprocessor = preprocessor if preprocessor is not None else Identity()

    def __call__(self, inputs):
        processed_inputs = self._preprocessor(inputs)
        self._handler(*processed_inputs)
