import unittest

from hamcrest import *

from kerosene.configs.configs import TrainerConfiguration, DatasetConfiguration
from kerosene.utils.configs import configs_to_html


class ConfigTests(unittest.TestCase):

    def setUp(self) -> None:
        self._config_1 = TrainerConfiguration({"num_epochs": 30, "batch_size": 32})
        self._config_2 = DatasetConfiguration({"path": "/home/data", "validation_split": 0.2})

    def test_should_produce_html_from_configs(self):
        config_html = configs_to_html([self._config_1, self._config_2])
        assert_that(config_html[0],
                    is_('<h2>Training Configuration</h2> \n <p>num_epochs: 30</p>\n<p>batch_size: 32</p>'))
        assert_that(config_html[1],
                    is_('<h2>Dataset Configuration</h2> \n <p>path: /home/data</p>\n<p>validation_split: 0.2</p>'))
