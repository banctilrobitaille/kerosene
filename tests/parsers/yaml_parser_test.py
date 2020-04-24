import unittest

import torch
from hamcrest import *

from kerosene.configs.parsers import CustomYamlParser


class YamlParserTest(unittest.TestCase):

    def setUp(self) -> None:
        self._path = "tests/parsers/test_yaml.yml"
        self._parser = CustomYamlParser()

    def test_should_parse_file_and_create_tensor(self):
        with open(self._path) as file:
            config = self._parser.safe_load(file)

        assert_that(config["weights"], instance_of(torch.Tensor))
        assert_that(config["tuple"], instance_of(tuple))
