import yaml
import torch


class Tensor(yaml.YAMLObject):
    yaml_tag = u'!pytorch/tensor'

    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def tensor(self):
        return self._tensor

    @classmethod
    def from_yaml(cls, loader, node):
        value = loader.construct_sequence(node, deep=True)
        tensor = torch.Tensor().new_tensor(value)
        return cls(tensor)


class KeroseneParser(object):

    def __init__(self):
        yaml.SafeLoader.add_constructor('!pytorch/tensor', Tensor.from_yaml)
        yaml.add_constructor('!pytorch/tensor', Tensor.from_yaml)

    @staticmethod
    def parse(file):
        with open(file, "r") as file:
            return yaml.safe_load(file)
