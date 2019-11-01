import torch
import yaml


class CustomYamlParser(object):

    def __init__(self):
        yaml.SafeLoader.add_constructor(u"!torch/tensor", CustomYamlParser.parse_tensor)
        yaml.SafeLoader.add_constructor(u"!python/tuple", CustomYamlParser.parse_tuple)

    @staticmethod
    def safe_load(file):
        return yaml.safe_load(file)

    @staticmethod
    def parse_tensor(loader, node):
        value = loader.construct_sequence(node, deep=True)
        tensor = torch.Tensor().new_tensor(value)
        return tensor

    @staticmethod
    def parse_tuple(loader, node):
        value = loader.construct_sequence(node, deep=True)
        tuple_ = tuple(value)
        return tuple_
