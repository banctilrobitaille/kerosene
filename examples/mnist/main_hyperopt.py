import logging

import torchvision
from hyperopt import fmin, tpe, hp
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from examples.mnist.models import SimpleConvNet
from kerosene.configs.configs import RunConfiguration
from kerosene.configs.parsers import YamlConfigurationParser
from kerosene.events.handlers.console import PrintTrainingStatus
from kerosene.events.handlers.visdom import PlotMonitors, PlotAvgGradientPerLayer
from kerosene.loggers.visdom.config import VisdomConfiguration
from kerosene.loggers.visdom.visdom import VisdomLogger
from kerosene.training.events import Event
from kerosene.training.trainers import ModelTrainerFactory, SimpleTrainer


def objective(hyper_params):
    model_config, training_config = YamlConfigurationParser.parse(CONFIG_FILE_PATH)

    train_loader = DataLoader(torchvision.datasets.MNIST('./files/', train=True, download=True, transform=Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))])), batch_size=training_config.batch_size_train, shuffle=True)

    test_loader = DataLoader(torchvision.datasets.MNIST('./files/', train=False, download=True, transform=Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))])), batch_size=training_config.batch_size_valid, shuffle=True)

    visdom_logger = VisdomLogger(VisdomConfiguration.from_yml(CONFIG_FILE_PATH))

    # Initialize the model trainers
    model_trainer = ModelTrainerFactory(model=SimpleConvNet()).create(model_config)

    # Train with the training strategy
    SimpleTrainer("MNIST Trainer", train_loader, test_loader, None, model_trainer, RunConfiguration(use_amp=False)) \
        .with_event_handler(PlotMonitors(every=500, visdom_logger=visdom_logger), Event.ON_BATCH_END) \
        .with_event_handler(PlotMonitors(visdom_logger=visdom_logger), Event.ON_EPOCH_END) \
        .with_event_handler(PlotAvgGradientPerLayer(every=500, visdom_logger=visdom_logger), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(PrintTrainingStatus(every=100), Event.ON_BATCH_END) \
        .train(training_config.nb_epochs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    CONFIG_FILE_PATH = "config.yml"

    search_space = {
        'SimpleNet': {'scheduler': {'params': {'lr': hp.normal('lr', 0, 2)}}}
    }

    best = fmin(objective, space=search_space, algo=tpe.suggest, max_evals=10)
