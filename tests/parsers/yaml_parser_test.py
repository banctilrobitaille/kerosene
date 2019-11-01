import yaml
import unittest
import torch
from hamcrest import *

from kerosene.parsers.yaml_parser import Tensor, YamlParser


class TestYamlParser(unittest.TestCase):

    def setUp(self) -> None:
        self._path = "parsers/test_yaml.yml"

    def test_should_parse_file_and_create_tensor(self):
        yaml.SafeLoader.add_constructor('!pytorch/tensor', Tensor.from_yaml)
        yaml.add_constructor('!pytorch/tensor', Tensor.from_yaml)

        file = open(self._path)
        data = yaml.safe_load(file)

        assert_that(data["weights"].tensor, instance_of(torch.Tensor))


class TestKeroseneParser(unittest.TestCase):
    def setUp(self) -> None:
        self._path = "parsers/test_yaml.yml"
        self._parser = YamlParser()

    def test_should_parse_file(self):
        with open(self._path) as file:
            config = self._parser.load(file)
        assert_that(config, instance_of(dict))
        assert_that(config["weights"].tensor, instance_of(torch.Tensor))
