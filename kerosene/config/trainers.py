class TrainingConfiguration(object):
    def __init__(self, config_dict):
        for key in config_dict:
            setattr(self, key, config_dict[key])


class ModelTrainerConfiguration(object):

    def __init__(self, model_name, model_type, model_params, optimizer_type, optimizer_params, scheduler_type,
                 scheduler_params, criterion_type, criterion_params, metric_type, metric_params):
        self._model_name = model_name
        self._model_type = model_type
        self._model_params = model_params

        self._optimizer_type = optimizer_type
        self._optimizer_params = optimizer_params

        self._scheduler_type = scheduler_type
        self._scheduler_params = scheduler_params

        self._criterion_type = criterion_type
        self._criterion_params = criterion_params

        self._metric_type = metric_type
        self._metric_params = metric_params

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

    @property
    def criterion_type(self):
        return self._criterion_type

    @property
    def criterion_params(self):
        return self._criterion_params

    @property
    def metric_type(self):
        return self._metric_type

    @property
    def metric_params(self):
        return self._metric_params

    @classmethod
    def from_dict(cls, model_name, config_dict):
        return cls(model_name, config_dict["type"], config_dict["params"], config_dict["optimizer"]["type"],
                   config_dict["optimizer"]["params"], config_dict["scheduler"]["type"],
                   config_dict["scheduler"]["params"], config_dict["criterion"]["type"],
                   config_dict["criterion"]["params"], config_dict["metric"]["type"],
                   config_dict["metric"]["params"])
