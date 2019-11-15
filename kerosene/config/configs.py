import abc

from kerosene.config.exceptions import InvalidConfigurationError


class Configuration(object):

    @abc.abstractmethod
    def to_html(self):
        raise NotImplementedError()


class CheckpointConfiguration(Configuration):
    def __init__(self, model_name, model_path, optimizer_path):
        self._model_name = model_name
        self._model_path = model_path
        self._optimizer_path = optimizer_path

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_path(self):
        return self._model_path

    @property
    def optimizer_path(self):
        return self._optimizer_path

    @classmethod
    def from_dict(cls, model_name, config_dict):
        try:
            return cls(model_name, config_dict.get("model_path"), config_dict.get("optimizer_path"))
        except KeyError as e:
            raise InvalidConfigurationError(
                "The provided checkpoint configuration is invalid. The section {} is missing.".format(e))

    def to_html(self):
        configuration_values = '\n'.join("<p>%s: %s</p>" % item for item in vars(self).items())
        return "<h2>Checkpoint Configuration</h2> \n {}".format(configuration_values)
