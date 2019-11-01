import unittest

import torch
from hamcrest import *

from kerosene.parsers.yaml_parser import YamlParser


class YamlParserTest(unittest.TestCase):

    def setUp(self) -> None:
        self._path = "test_yaml.yml"
        self._parser = YamlParser()

    def test_should_parse_file_and_create_tensor(self):
        with open(self._path) as file:
            config = self._parser.safe_load(file)

        assert_that(config["weights"], instance_of(torch.Tensor))
        assert_that(config["list"], instance_of(list))
        assert_that(config["tuple"], instance_of(tuple))
