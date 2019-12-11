import os
import unittest

import torch
from hamcrest import *

from kerosene.configs.parsers import YamlConfigurationParser
from kerosene.training.trainers import ModelTrainerFactory

from kerosene.optim.optimizers import OptimizerFactory


class ModelTrainerFactoryTest(unittest.TestCase):
    CHECKPOINT_PATH = "saves/checkpoint"

    def setUp(self) -> None:
        if not os.path.exists(self.CHECKPOINT_PATH):
            os.makedirs(self.CHECKPOINT_PATH)

        if not os.path.exists(self.CHECKPOINT_PATH):
            os.makedirs(self.CHECKPOINT_PATH)

        model_trainer_config, _ = YamlConfigurationParser.parse("tests/configs/valid_config.yml")

        torch.manual_seed(42)

        self._model = torch.nn.Linear(32, 2)
        self._inputs = torch.rand((1, 32))
        self._truth = torch.ones((1,), dtype=torch.long)
        self._optims = [OptimizerFactory().create(model_trainer_config.optimizer_types[0], self._model,
                                                  **model_trainer_config.optimizer_params[0]),
                        OptimizerFactory().create(model_trainer_config.optimizer_types[1], self._model,
                                                  **model_trainer_config.optimizer_params[1])]
        self._epoch = 10

        self._model_trainer = ModelTrainerFactory(model=self._model).create(model_trainer_config)
        self._model_trainer._use_amp = False

        out = self._model_trainer.forward(self._inputs)
        loss = self._model_trainer.compute_loss(out, self._truth)
        loss.backward()
        self._model_trainer.step()

        torch.save({
            "epoch_num": self._epoch,
            "model_state_dict": self._model_trainer.model_state,
            "optimizer_state_dict": self._model_trainer.optimizer_states
        }, os.path.join(self.CHECKPOINT_PATH, "checkpoint.tar"))

    def test_should_reload_state_dict_and_have_same_state(self):
        model_trainer_config, _ = YamlConfigurationParser.parse("tests/configs/valid_config_with_checkpoint.yml")

        model_trainer = ModelTrainerFactory(model=self._model).create(model_trainer_config)
        model_trainer._use_amp = False

        out = model_trainer.forward(self._inputs)
        loss = model_trainer.compute_loss(out, self._truth)
        loss.backward()
        model_trainer.step()

        assert_that(model_trainer.model.state_dict()["weight"].cpu().numpy().all(),
                    equal_to(self._model.state_dict()["weight"].cpu().numpy().all()))
        assert_that(model_trainer.model.state_dict()["bias"].cpu().numpy().all(),
                    equal_to(self._model.state_dict()["bias"].cpu().numpy().all()))
        [assert_that(optimizer.state_dict()["state"], is_(optim.state_dict()["state"])) for optimizer, optim in
         zip(list(self._model_trainer.optimizers.values()), self._optims)]
        [assert_that(optimizer.state_dict()["param_groups"][0], equal_to(optim.state_dict()["param_groups"][0]))
         for optimizer, optim in zip(list(self._model_trainer.optimizers.values()), self._optims)]
