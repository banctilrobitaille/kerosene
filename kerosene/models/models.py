from abc import ABC, abstractmethod


class ModelFactory(ABC):
    @abstractmethod
    def create(self, model_type, **params):
        raise NotImplementedError
