import logging
import os
import sys
from argparse import ArgumentParser

from kerosene.config.trainers import RunConfiguration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../../../')
import torch
import torchvision
import multiprocessing
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from kerosene.config.parsers import YamlConfigurationParser
from kerosene.events import Event
from kerosene.events.handlers.console import ConsoleLogger
from kerosene.events.handlers.visdom.config import VisdomConfiguration
from kerosene.events.handlers.visdom.visdom import VisdomLogger
from kerosene.events.preprocessors.visdom import PlotAllModelStateVariables
from kerosene.training.trainers import ModelTrainerFactory
from tests.functionals.distributed.models import SimpleNet
from tests.functionals.distributed.mnist_trainer import MNISTTrainer



class ArgsParserFactory(object):

    @staticmethod
    def create_parser():
        parser = ArgumentParser(description='DeepNormalize Training')
        parser.add_argument("--local_rank", dest="local_rank", default=0, type=int, help="The local_rank of the GPU.")
        parser.add_argument('--distributed', action='store_true', default=False)
        return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    CONFIG_FILE_PATH = "config.yml"
    args = ArgsParserFactory.create_parser().parse_args()
    run_config = RunConfiguration(True, "O2", args.local_rank, args.distributed)
    print("local_rank : {} ".format(args.local_rank))
    print(run_config.device)
    
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    model_trainer_config, training_config = YamlConfigurationParser.parse(CONFIG_FILE_PATH)

    train_dataset = torchvision.datasets.MNIST('./files/', train=True, download=True, transform=Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))]))

    test_dataset = torchvision.datasets.MNIST('./files/', train=False, download=True, transform=Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))]))

    if run_config.is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=training_config.batch_size,
                                               shuffle=False if run_config.is_distributed else True,
                                               num_workers=multiprocessing.cpu_count(),
                                               sampler=train_sampler if run_config.is_distributed else None,
                                               pin_memory=torch.cuda.is_available())

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=training_config.batch_size,
                                              shuffle=False if run_config.is_distributed else True,
                                              num_workers=multiprocessing.cpu_count(),
                                              sampler=valid_sampler if run_config.is_distributed else None,
                                              pin_memory=torch.cuda.is_available())
    if run_config.local_rank == 0:
        # Initialize the loggers
        visdom_logger = VisdomLogger(VisdomConfiguration.from_yml(CONFIG_FILE_PATH))

    # Initialize the model trainers
    model_trainer = ModelTrainerFactory(model=SimpleNet()).create(model_trainer_config)

    if run_config.local_rank == 0:
        # Train with the training strategy
        trainer = MNISTTrainer(training_config, model_trainer, train_loader, test_loader, run_config) \
            .with_event_handler(ConsoleLogger(), Event.ON_EPOCH_END) \
            .with_event_handler(visdom_logger, Event.ON_EPOCH_END, PlotAllModelStateVariables()) \
            .train(training_config.nb_epochs)
    else:
        trainer = MNISTTrainer(training_config, model_trainer, train_loader, test_loader, run_config) \
            .with_event_handler(ConsoleLogger(), Event.ON_EPOCH_END) \
            .train(training_config.nb_epochs)
