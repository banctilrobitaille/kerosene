import unittest
import torch
from kerosene.config.trainers import RunConfiguration

from kerosene.training.trainers import ModelTrainer, SimpleTrainer
from mockito import mock
from kerosene.config.configs import CheckpointConfiguration
import os
from torch.utils.data.dataloader import DataLoader
from kerosene.metrics.metrics import Accuracy
from hamcrest import *


class TrainerTest(unittest.TestCase):
    MODEL_PATH = "saves/model"
    OPTIMIZER_PATH = "saves/optimizer"

    def setUp(self) -> None:
        if not os.path.exists(self.MODEL_PATH):
            os.makedirs(self.MODEL_PATH)

        if not os.path.exists(self.OPTIMIZER_PATH):
            os.makedirs(self.OPTIMIZER_PATH)

        self._model = torch.nn.Conv2d(32, 32, (3, 3), (1, 1))
        self._optim = torch.optim.SGD(self._model.parameters(), lr=0.01, momentum=0.9)
        self._epoch = 10
        self._model_trainer = ModelTrainer("test_model_trainer", self._model, mock(torch.nn.CrossEntropyLoss),
                                           self._optim, mock(torch.optim.lr_scheduler), mock(Accuracy))
        self._run_config = RunConfiguration()
        torch.save({
            "epoch_num": self._epoch,
            "model_state_dict": self._model.state_dict()
        }, os.path.join(self.MODEL_PATH, "model.tar"))
        torch.save({
            "optimizer_state_dict": self._optim.state_dict()
        }, os.path.join(self.OPTIMIZER_PATH, "optimizer.tar"))

    def test_should_reload_state_dict_and_have_same_state(self):
        checkpoint_config = CheckpointConfiguration("test_model_trainer", os.path.join(self.MODEL_PATH, "model.tar"),
                                                    os.path.join(self.OPTIMIZER_PATH, "optimizer.tar"))

        self._trainer = SimpleTrainer("test_trainer", mock(DataLoader), mock(DataLoader), [self._model_trainer],
                                      self._run_config)

        self._trainer.load([checkpoint_config])

        assert_that(self._trainer.model_trainers[0].model.state_dict()["weight"].cpu().numpy().all(),
                    equal_to(self._model.state_dict()["weight"].cpu().numpy().all()))
        assert_that(self._trainer.model_trainers[0].model.state_dict()["bias"].cpu().numpy().all(),
                    equal_to(self._model.state_dict()["bias"].cpu().numpy().all()))
        assert_that(self._trainer.model_trainers[0].optimizer.state_dict()["state"],
                    is_(self._optim.state_dict()["state"]))
        assert_that(self._trainer.model_trainers[0].optimizer.state_dict()["param_groups"],
                    equal_to(self._optim.state_dict()["param_groups"]))
        assert_that(self._trainer.epoch, is_(self._epoch))
