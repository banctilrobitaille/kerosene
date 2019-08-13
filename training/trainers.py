import torch
from blinker import signal
from ignite.metrics import Metric
from torch import nn
from torch.utils.data import DataLoader

from metrics.gauges import AverageGauge
from training.state import TrainerState
from training.strategies import TrainingStrategy


class Trainer(object):
    ON_TRAINING_BEGIN = signal("OnTrainingBegin")
    ON_TRAINING_END = signal("OnTrainingEnd")
    ON_EPOCH_BEGIN = signal("OnEpochBegin")
    ON_TRAIN_EPOCH_BEGIN = signal("OnTrainEpochBegin")
    ON_TRAIN_EPOCH_END = signal("OnTrainEpochEnd")
    ON_VALID_EPOCH_BEGIN = signal("OnValidEpochBegin")
    ON_VALID_EPOCH_END = signal("OnValidEpochEnd")
    ON_EPOCH_END = signal("OnEpochEnd")
    ON_TRAIN_BATCH_BEGIN = signal("OnTrainBatchBegin")
    ON_TRAIN_BATCH_END = signal("OnTrainBatchEnd")
    ON_VALID_BATCH_BEGIN = signal("OnValidBatchBegin")
    ON_VALID_BATCH_END = signal("OnValidBatchEnd")

    def __init__(self, name, train_data_loader: DataLoader, valid_data_loader: DataLoader,
                 train_strategy: TrainingStrategy, configuration):

        super().__init__(name)
        self._train_data_loader = train_data_loader
        self._valid_data_loader = valid_data_loader
        self._train_strategy = train_strategy
        self._state = TrainerState()

        self._nb_epochs = configuration.nb_epochs
        self._current_train_batch = 0
        self._current_valid_batch = 0
        self._current_epoch = 0

    @property
    def state(self):
        return self._state

    @property
    def nb_epochs(self):
        return self._nb_epochs

    @property
    def nb_of_train_batch(self):
        return len(self._train_data_loader)

    @property
    def nb_of_valid_batch(self):
        return len(self._valid_data_loader)

    @property
    def global_train_step(self):
        return self._current_epoch * len(self._train_data_loader) + self._current_train_batch

    @property
    def global_valid_step(self):
        return self._current_epoch * len(self._valid_data_loader) + self._current_valid_batch

    def train(self):
        self.ON_TRAINING_BEGIN.send(self)

        for self._current_epoch in range(0, self._nb_epochs):
            self.ON_EPOCH_BEGIN.send()
            self.ON_TRAIN_EPOCH_BEGIN.send()
            self._train_epoch()
            self.ON_TRAIN_EPOCH_END.send()
            self.ON_VALID_EPOCH_BEGIN.send()
            self._validate_epoch()
            self.ON_VALID_EPOCH_END.send()
            self.ON_EPOCH_END.send()

        self.ON_TRAINING_END.send(self)

    def _train_epoch(self):

        for self._current_train_batch, (inputs, target) in enumerate(self._train_data_loader):
            self.ON_TRAIN_BATCH_BEGIN.send(self)
            target = target.cuda(non_blocking=True)
            self._train_strategy.train_step(inputs, target)
            self.ON_TRAIN_BATCH_END.send(self)

    def _validate_epoch(self):

        with torch.no_grad():
            for self._current_valid_batch, (inputs, target) in enumerate(self._valid_data_loader):
                self.ON_VALID_BATCH_BEGIN.send(self)
                target = target.cuda(non_blocking=True)
                self._train_strategy.validate_step(inputs, target)
                self.ON_VALID_BATCH_END.send(self)


class ModelTrainer(nn.Module):

    def __init__(self, model_name, model, criterion, optimizer, scheduler, metric_computer: Metric):
        super().__init__()
        self._model_name = model_name
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._metric_computer = metric_computer

        self._train_loss = AverageGauge()
        self._valid_loss = AverageGauge()

    @property
    def optimizer_state(self):
        return self._optimizer.state_dict()

    @property
    def model_state(self):
        return self._model.state_dict()

    @property
    def criterion(self):
        return self._criterion

    @property
    def scheduler(self):
        return self._scheduler

    def forward(self, *input):
        return self._model.forward(*input)

    def eval(self):
        self._model.eval()

    def train(self, mode=True):
        self._model.train()

    def step(self):
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def compute_train_loss(self, pred, target):
        loss = self._criterion(pred, target)

        self._train_loss.update(loss)

        return loss

    def compute_valid_loss(self, pred, target):
        loss = self._criterion(pred, target)
        self._valid_loss.update(loss)

    def update_metric(self, pred, target):
        return self._metric_computer.update((pred, target))

    def _on_train_epoch_begin(self, state: TrainerState):
        self._metric_computer.reset()
        self._train_loss_gauge.reset()

    def _on_valid_epoch_begin(self, state: TrainerState):
        self._metric_computer.reset()
        self._valid_loss_gauge.reset()

    def on_epoch_end(self, state: TrainerState):
        self._train_metric = self._metric_computer.compute()

    def on_train_batch_end(self, state: TrainerState):
        self._criterion.reset()

    def on_valid_batch_end(self, state: TrainerState):
        pass
