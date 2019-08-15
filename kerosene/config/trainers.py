class TrainingConfiguration(object):
    def __init__(self, config_dict):
        for key in config_dict:
            setattr(self, key, config_dict[key])

    @classmethod
    def from_dict(cls, config_dict):
        pass


class ModelTrainerConfiguration(object):

    def __init__(self, model_name, model_type, model_params, optimizer_type, optimizer_params, scheduler_type,
                 scheduler_params):
        self._model_name = model_name
        self._model_type = model_type
        self._model_params = model_params

        self._optimizer_type = optimizer_type
        self._optimizer_params = optimizer_params

        self._scheduler_type = scheduler_type
        self._scheduler_params = scheduler_params

    @property
    def model_name(self):
        return self._model_name

    @property
    def type(self):
        return self._model_type

    @property
    def model_params(self):
        return self._model_params

    @property
    def optimizer_type(self):
        return self._optimizer_type

    @property
    def optimizer_params(self):
        return self._optimizer_params

    @property
    def scheduler_type(self):
        return self._scheduler_type

    @property
    def scheduler_params(self):
        return self._scheduler_params

    @classmethod
    def from_dict(cls, model_name, config_dict):
        cls(model_name, config_dict["type"], config_dict["params"], config_dict["optimizer"]["type"],
            config_dict["optimizer"]["params"], config_dict["scheduler"]["type"], config_dict["scheduler"]["params"])
