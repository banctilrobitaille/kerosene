class TrainerState(object):
    def __init__(self, epoch=0, train_step=0, valid_step=0, train_data_loader=None,
                 valid_data_loader=None, model_trainer_states=()):
        self._epoch = epoch
        self._train_step = train_step
        self._valid_step = valid_step
        self._train_data_loader = train_data_loader
        self._valid_data_loader = valid_data_loader

        self._model_trainer_states = model_trainer_states

    @property
    def epoch(self):
        return self._epoch

    @property
    def train_step(self):
        return self._train_step

    @property
    def valid_step(self):
        return self._valid_step

    @property
    def train_data_loader(self):
        return self._train_data_loader

    @property
    def valid_data_loader(self):
        return self._valid_data_loader

    @property
    def model_trainer_states(self):
        return self._model_trainer_states

    def __str__(self):
        return "Training state: Epoch: {} | Training step: {} | Validation step: {} \n {}".format(
            self._epoch, self._train_step, self._valid_step, "\n".join(self._model_trainer_states))


class ModelTrainerState(object):
    def __init__(self, name, train_loss=0, valid_loss=0, train_metric=0, valid_metric=0, model_state=None,
                 optimizer_state=None):
        self._name = name
        self._train_loss = train_loss
        self._valid_loss = valid_loss
        self._train_metric = train_metric
        self._valid_metric = valid_metric
        self._model_state = model_state
        self._optimizer_state = optimizer_state

    @property
    def name(self):
        return self._name

    @property
    def train_loss(self):
        return self._train_loss

    @property
    def valid_loss(self):
        return self._valid_loss

    @property
    def train_metric(self):
        return self._train_metric

    @property
    def valid_metric(self):
        return self._valid_metric

    @property
    def model_state(self):
        return self._model_state

    @property
    def optimizer_state(self):
        return self._optimizer_state

    def __str__(self):
        return "{} -- Training loss: {} | Validation loss: {} | Training metric: {} | Validation metric: {}".format(
            self._name, self._train_loss, self._valid_loss, self._train_metric, self._valid_metric)
