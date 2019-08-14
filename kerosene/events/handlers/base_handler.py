from abc import ABC, abstractmethod


class EventHandler(ABC):

    @abstractmethod
    def __call__(self, *inputs):
        raise NotImplementedError()
