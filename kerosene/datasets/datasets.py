
from abc import ABC, abstractmethod

class DatasetFactory(ABC):

    def __init__(self, dataset_config):
        self._dataset_config = dataset_config

    @abstractmethod
    def create_train_valid_test_datasets(self):
        raise NotImplementedError()