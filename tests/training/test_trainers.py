import os
import unittest

import torch
from hamcrest import *

from kerosene.configs.parsers import YamlConfigurationParser
from kerosene.training.trainers import ModelTrainerFactory


class ModelTrainerFactoryTest(unittest.TestCase):
    CHECKPOINT_PATH = "saves/checkpoint"

    def setUp(self) -> None:
        if not os.path.exists(self.CHECKPOINT_PATH):
            os.makedirs(self.CHECKPOINT_PATH)

        if not os.path.exists(self.CHECKPOINT_PATH):
            os.makedirs(self.CHECKPOINT_PATH)

        model_trainer_config, _ = YamlConfigurationParser.parse("tests/training/valid_config.yml")

        torch.manual_seed(42)

        self._model = torch.nn.Linear(32, 2)
        self._inputs = torch.rand((1, 32))
        self._truth = torch.ones((1,), dtype=torch.long)
        self._optim = torch.optim.Adam(self._model.parameters(), lr=0.01)
        self._epoch = 10

        self._model_trainer = ModelTrainerFactory(model=self._model).create(model_trainer_config)
        self._model_trainer[0]._use_amp = False

        out = self._model_trainer[0].forward(self._inputs)
        loss = self._model_trainer[0].compute_loss(out, self._truth)
        loss.backward()
        self._model_trainer[0].step()

        torch.save({
            "epoch_num": self._epoch,
            "model_state_dict": self._model_trainer[0].model_state,
            "optimizer_state_dict": self._model_trainer[0].optimizer_states
        }, os.path.join(self.CHECKPOINT_PATH, "checkpoint.tar"))

    def test_should_reload_state_dict_and_have_same_state(self):
        model_trainer_config, _ = YamlConfigurationParser.parse("tests/training/valid_config_with_checkpoint.yml")

        model_trainer = ModelTrainerFactory(model=self._model).create(model_trainer_config)
        model_trainer[0]._use_amp = False

        out = model_trainer[0].forward(self._inputs)
        loss = model_trainer[0].compute_loss(out, self._truth)
        loss.backward()
        model_trainer[0].step()

        assert_that(model_trainer[0].model.state_dict()["weight"].cpu().numpy().all(),
                    equal_to(self._model.state_dict()["weight"].cpu().numpy().all()))
        assert_that(model_trainer[0].model.state_dict()["bias"].cpu().numpy().all(),
                    equal_to(self._model.state_dict()["bias"].cpu().numpy().all()))
        assert_that(model_trainer[0].optimizers[0].state_dict()["state"],
                    is_(self._optim.state_dict()["state"]))
        assert_that(model_trainer[0].optimizers[0].state_dict()["param_groups"][0],
                    equal_to(self._optim.state_dict()["param_groups"][0]))
