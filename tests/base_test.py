import unittest

from ignite.metrics import Accuracy, Recall
from mockito import mock, spy
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader

from kerosene.nn.utils.gradients import GradientClippingStrategy
from kerosene.training.trainers import Model


class BaseTestEnvironment(unittest.TestCase):
    MODEL_NAME = "Harry Potter"
    TRAINER_NAME = "Drago Malfoy"

    def setUp(self) -> None:
        self._model_mock = mock(nn.Module)
        self._criterion_mock = spy(nn.CrossEntropyLoss())
        self._optimizer_mock = mock(Optimizer)
        self._scheduler_mock = mock(lr_scheduler)
        self._accuracy_computer_mock = spy(Accuracy())
        self._recall_computer_mock = spy(Recall())
        self._gradient_clipping_strategy = mock(GradientClippingStrategy)

        self._training_data_loader_mock = mock(DataLoader)
        self._valid_data_loader_mock = mock(DataLoader)
        self._test_data_loader_mock = mock(DataLoader)

        self._model_trainer = Model(self.MODEL_NAME, self._model_mock, self._criterion_mock,
                                    self._optimizer_mock, self._scheduler_mock,
                                    {"Accuracy": self._accuracy_computer_mock,
                                            "Recall": self._recall_computer_mock}, self._gradient_clipping_strategy)
