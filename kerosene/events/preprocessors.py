from abc import ABC, abstractmethod


class EventPreprocessor(ABC):

    @abstractmethod
    def __call__(self, state):
        raise NotImplementedError()


class Identity(EventPreprocessor):

    def __call__(self, state):
        return state
