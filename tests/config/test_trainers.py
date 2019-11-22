import unittest

from hamcrest import *

from kerosene.config.exceptions import InvalidConfigurationError
from kerosene.config.parsers import YamlConfigurationParser
from kerosene.config.configs import ModelTrainerConfiguration


class TestModelTrainerConfiguration(unittest.TestCase):
    VALID_CONFIG_FILE_PATH = "tests/config/valid_config.yml"
    INVALID_CONFIG_FILE_PATH = "tests/config/invalid_config.yml"

    MODELS_CONFIG_YML_TAG = "models"

    SIMPLE_NET_NAME = "SimpleNet"
    SIMPLE_NET_TYPE = "SimpleNet"

    SIMPLE_NET_OPTIMIZER_TYPE = "SGD"
    SIMPLE_NET_OPTIMIZER_PARAMS = {'lr': 0.01, 'momentum': 0.5, 'weight_decay': 0}

    SIMPLE_NET_SCHEDULER_TYPE = "ReduceLROnPlateau"
    SIMPLE_NET_SCHEDULER_PARAMS = {'mode': 'min', 'factor': 0.1, 'patience': 3}

    SIMPLE_NET_CRITERION_TYPE = "CrossEntropyLoss"
    SIMPLE_NET_METRIC_TYPE = "Accuracy"

    SIMPLE_NET_GRADIENT_CLIPPING = {"clipping_strategy": "norm", "params": {"max_norm": 1.0}}

    def test_should_parse_valid_model_trainer_config(self):
        expected_config_dict = {self.SIMPLE_NET_NAME: {'type': self.SIMPLE_NET_TYPE,
                                                       'optimizer': {'type': self.SIMPLE_NET_OPTIMIZER_TYPE,
                                                                     'params': self.SIMPLE_NET_OPTIMIZER_PARAMS},
                                                       'scheduler': {'type': self.SIMPLE_NET_SCHEDULER_TYPE,
                                                                     'params': self.SIMPLE_NET_SCHEDULER_PARAMS},
                                                       'criterion': {'type': self.SIMPLE_NET_CRITERION_TYPE},
                                                       'metric': {'type': self.SIMPLE_NET_METRIC_TYPE},
                                                       'gradients': self.SIMPLE_NET_GRADIENT_CLIPPING}}
        config_dict = YamlConfigurationParser.parse_section(self.VALID_CONFIG_FILE_PATH, self.MODELS_CONFIG_YML_TAG)
        model_trainer_config = ModelTrainerConfiguration.from_dict(self.SIMPLE_NET_NAME,
                                                                   config_dict[self.SIMPLE_NET_NAME])

        assert_that(config_dict, equal_to(expected_config_dict))

        assert_that(model_trainer_config.optimizer_type, equal_to(self.SIMPLE_NET_OPTIMIZER_TYPE))
        assert_that(model_trainer_config.optimizer_params, equal_to(self.SIMPLE_NET_OPTIMIZER_PARAMS))
        assert_that(model_trainer_config.scheduler_type, equal_to(self.SIMPLE_NET_SCHEDULER_TYPE))
        assert_that(model_trainer_config.scheduler_params, equal_to(self.SIMPLE_NET_SCHEDULER_PARAMS))
        assert_that(model_trainer_config.criterion_type, equal_to(self.SIMPLE_NET_CRITERION_TYPE))
        assert_that(model_trainer_config.criterion_params, equal_to(None))
        assert_that(model_trainer_config.metric_type, equal_to(self.SIMPLE_NET_METRIC_TYPE))
        assert_that(model_trainer_config.metric_params, equal_to(None))

    def test_should_throw_on_invalid_model_trainer_config(self):
        config_dict = YamlConfigurationParser.parse_section(self.INVALID_CONFIG_FILE_PATH, self.MODELS_CONFIG_YML_TAG)

        assert_that(calling(ModelTrainerConfiguration.from_dict).with_args(self.SIMPLE_NET_NAME,
                                                                           config_dict[self.SIMPLE_NET_NAME]),
                    raises(InvalidConfigurationError))
