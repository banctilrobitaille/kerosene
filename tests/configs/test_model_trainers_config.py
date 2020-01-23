import unittest

from hamcrest import *

from kerosene.configs.configs import ModelConfiguration
from kerosene.configs.exceptions import InvalidConfigurationError
from kerosene.configs.parsers import YamlConfigurationParser


class TestModelTrainerConfiguration(unittest.TestCase):
    VALID_CONFIG_FILE_PATH = "tests/configs/valid_config.yml"
    INVALID_CONFIG_FILE_PATH = "tests/configs/invalid_config.yml"

    MODELS_CONFIG_YML_TAG = "models"

    SIMPLE_NET_NAME = "SimpleNet"
    SIMPLE_NET_TYPE = "SimpleNet"

    SIMPLE_NET_OPTIMIZER_TYPE = "SGD"
    SIMPLE_NET_OPTIMIZER_PARAMS = {'lr': 0.01, 'momentum': 0.5, 'weight_decay': 0}

    SIMPLE_NET_SCHEDULER_TYPE = "ReduceLROnPlateau"
    SIMPLE_NET_SCHEDULER_PARAMS = {'mode': 'min', 'factor': 0.1, 'patience': 3}

    SIMPLE_NET_CRITERION_TYPE = "L1Loss"
    SIMPLE_NET_CRITERION_TYPE_2 = "MSELoss"

    SIMPLE_NET_METRIC_TYPE_1 = "Dice"
    SIMPLE_NET_METRIC_PARAMS_1 = {"num_classes": 4, "reduction": None, "ignore_index": 0, "average": None,
                                  "weight": None}

    SIMPLE_NET_METRIC_TYPE_2 = "Accuracy"

    SIMPLE_NET_GRADIENT_CLIPPING = {"clipping_strategy": "norm", "params": {"max_norm": 1.0}}

    def test_should_parse_valid_model_trainer_config(self):
        expected_config_dict = {self.SIMPLE_NET_NAME: {'type': self.SIMPLE_NET_TYPE,
                                                       'optimizer': {'type': self.SIMPLE_NET_OPTIMIZER_TYPE,
                                                                     'params': self.SIMPLE_NET_OPTIMIZER_PARAMS},
                                                       'scheduler': {'type': self.SIMPLE_NET_SCHEDULER_TYPE,
                                                                     'params': self.SIMPLE_NET_SCHEDULER_PARAMS},
                                                       'criterion': {"cycle": {'type': self.SIMPLE_NET_CRITERION_TYPE},
                                                                     "gan": {'type': self.SIMPLE_NET_CRITERION_TYPE_2}},
                                                       'metrics': {'Dice':{'type': self.SIMPLE_NET_METRIC_TYPE_1,
                                                                    'params': self.SIMPLE_NET_METRIC_PARAMS_1},
                                                                   'Accuracy': {'type': self.SIMPLE_NET_METRIC_TYPE_2}},
                                                       'gradients': self.SIMPLE_NET_GRADIENT_CLIPPING}}
        config_dict = YamlConfigurationParser.parse_section(self.VALID_CONFIG_FILE_PATH, self.MODELS_CONFIG_YML_TAG)
        model_trainer_config = ModelConfiguration.from_dict(self.SIMPLE_NET_NAME,
                                                            config_dict[self.SIMPLE_NET_NAME])

        assert_that(config_dict, equal_to(expected_config_dict))

        assert_that(model_trainer_config.optimizer_type, equal_to(self.SIMPLE_NET_OPTIMIZER_TYPE))
        assert_that(model_trainer_config.optimizer_params, equal_to(self.SIMPLE_NET_OPTIMIZER_PARAMS))
        assert_that(model_trainer_config.scheduler_type, equal_to(self.SIMPLE_NET_SCHEDULER_TYPE))
        assert_that(model_trainer_config.scheduler_params, equal_to(self.SIMPLE_NET_SCHEDULER_PARAMS))
        assert_that(model_trainer_config.criterions_configs[0].type, equal_to(self.SIMPLE_NET_CRITERION_TYPE))
        assert_that(model_trainer_config.criterions_configs[1].type, equal_to(self.SIMPLE_NET_CRITERION_TYPE_2))
        assert_that(model_trainer_config.metrics_configs[0].type, equal_to(self.SIMPLE_NET_METRIC_TYPE_1))
        assert_that(model_trainer_config.metrics_configs[0].params, equal_to(self.SIMPLE_NET_METRIC_PARAMS_1))

    def test_should_throw_on_invalid_model_trainer_config(self):
        config_dict = YamlConfigurationParser.parse_section(self.INVALID_CONFIG_FILE_PATH, self.MODELS_CONFIG_YML_TAG)

        assert_that(calling(ModelConfiguration.from_dict).with_args(self.SIMPLE_NET_NAME,
                                                                    config_dict[self.SIMPLE_NET_NAME]),
                    raises(InvalidConfigurationError))
