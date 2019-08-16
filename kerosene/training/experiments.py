from abc import ABC, abstractmethod
from typing import Tuple

from torch.utils.data import Dataset


class Experiment(ABC):
    def __init__(self, name):
        self._name = name

    @abstractmethod
    def prepare_data(self, *args, **kwargs) -> Tuple[Dataset]:
        raise NotImplementedError()
