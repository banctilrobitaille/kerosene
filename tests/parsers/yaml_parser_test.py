import yaml
import unittest
import torch
from hamcrest import *

from kerosene.parser.yaml_parser import Tensor, KeroseneParser


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
        self._parser = KeroseneParser()

    def test_should_parse_file(self):
        config = self._parser.parse(self._path)
        assert_that(config, instance_of(dict))
        assert_that(config["weights"].tensor, instance_of(torch.Tensor))
