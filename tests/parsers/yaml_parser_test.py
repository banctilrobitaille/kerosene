import yaml
import unittest
import torch
from hamcrest import *

from kerosene.parsers.yaml_parser import Tensor, YamlParser


class YamlParserTest(unittest.TestCase):

    def setUp(self) -> None:
        self._path = "parsers/test_yaml.yml"
        self._parser = YamlParser()

    def test_should_parse_file_and_create_tensor(self):
        yaml.SafeLoader.add_constructor('!pytorch/tensor', Tensor.from_yaml)
        yaml.add_constructor('!pytorch/tensor', Tensor.from_yaml)

        with open(self._path) as file:
            config = self._parser.safe_load(file)

        assert_that(config["weights"], instance_of(torch.Tensor))