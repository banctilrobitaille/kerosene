import yaml
import torch


class Tensor(yaml.YAMLObject):
    yaml_tag = u"!pytorch/tensor"
    yaml_loader = yaml.SafeLoader

    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def tensor(self):
        return self._tensor

    @classmethod
    def from_yaml(cls, loader, node):
        value = loader.construct_sequence(node, deep=True)
        tensor = torch.Tensor().new_tensor(value)
        return tensor


class Tuple(yaml.YAMLObject):
    yaml_tag = u"!python/tuple"
    yaml_loader = yaml.SafeLoader

    def __init__(self, tuple_):
        self._tuple = tuple_

    @property
    def tuple(self):
        return self._tuple

    @classmethod
    def from_yaml(cls, loader, node):
        value = loader.construct_sequence(node, deep=True)
        tuple_ = tuple(value)
        return tuple_


class List(yaml.YAMLObject):
    yaml_tag = u"!python/list"
    yaml_loader = yaml.SafeLoader

    def __init__(self, list_):
        self._list = list_

    @property
    def list(self):
        return self._list

    @classmethod
    def from_yaml(cls, loader, node):
        value = loader.construct_sequence(node, deep=True)
        list_ = list(value)
        return list_


class YamlParser(object):

    def __init__(self):
        yaml.SafeLoader.add_constructor(u"!pytorch/tensor", Tensor.from_yaml)
        yaml.add_constructor(u"!pytorch/tensor", Tensor.from_yaml, Loader=yaml.SafeLoader)

        yaml.SafeLoader.add_constructor(u"!python/tuple", Tuple.from_yaml)
        yaml.add_constructor(u"!python/tuple", Tuple.from_yaml, Loader=yaml.SafeLoader)

        yaml.SafeLoader.add_constructor(u"!python/list", Tuple.from_yaml)
        yaml.add_constructor(u"!python/list", Tuple.from_yaml, Loader=yaml.SafeLoader)

    @staticmethod
    def safe_load(file):
        return yaml.safe_load(file)
