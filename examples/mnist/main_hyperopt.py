import logging
import math

import torchvision
from hyperopt import fmin, tpe, hp, STATUS_OK
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from examples.mnist.models import SimpleConvNet
from kerosene.configs.configs import RunConfiguration
from kerosene.configs.parsers import YamlConfigurationParser
from kerosene.events import Monitor, Phase
from kerosene.events.handlers.console import PrintTrainingStatus
from kerosene.training.events import Event
from kerosene.training.trainers import ModelTrainerFactory, SimpleTrainer


def objective(hyper_params):
    # Update the trainer with the new hyper-parameters
    model_config.update(hyper_params)

    # Create the model trainer
    model_trainer = ModelTrainerFactory(model=SimpleConvNet()).create(model_config)

    # Train with the training strategy
    monitor = SimpleTrainer("MNIST Trainer", train_loader, valid_loader, None, model_trainer,
                            RunConfiguration(use_amp=False)) \
        .with_event_handler(PrintTrainingStatus(every=100), Event.ON_BATCH_END) \
        .train(training_config.nb_epochs)

    return {'loss': monitor["SimpleNet"][Phase.VALIDATION][Monitor.LOSS]["CrossEntropy"], 'status': STATUS_OK}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    CONFIG_FILE_PATH = "config.yml"

    model_config, training_config = YamlConfigurationParser.parse(CONFIG_FILE_PATH)

    train_loader = DataLoader(torchvision.datasets.MNIST('./files/', train=True, download=True, transform=Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))])), batch_size=training_config.batch_size_train, shuffle=True)

    valid_loader = DataLoader(torchvision.datasets.MNIST('./files/', train=False, download=True, transform=Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))])), batch_size=training_config.batch_size_valid, shuffle=True)

    search_space = {
        'SimpleNet': {'optimizer': {'params': {'lr': hp.loguniform('lr', math.log(0.0005), math.log(0.01))}}}
    }

    best = fmin(objective, space=search_space, algo=tpe.suggest, max_evals=2)

    print("The best hyper-parameters are: {}".format(best))
