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
from kerosene.utils.distributed import on_single_device


class ArgsParserFactory(object):

    @staticmethod
    def create_parser():
        parser = ArgumentParser(description='DeepNormalize Training')
        parser.add_argument("--use_amp", dest="use_amp", action="store_true", default=True)
        parser.add_argument("--amp-opt-level", dest="amp_opt_level", type=str, default="O2",
                            help="O0 - FP32 training, O1 - Mixed Precision (recommended), O2 - Almost FP16 Mixed Precision, O3 - FP16 Training.")
        parser.add_argument("--local_rank", dest="local_rank", default=0, type=int, help="The local_rank of the GPU.")
        return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    CONFIG_FILE_PATH = "config.yml"
    args = ArgsParserFactory.create_parser().parse_args()
    run_config = RunConfiguration(True, args.amp_opt_level, args.local_rank)
    devices = run_config.devices

    torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank)
    model_trainer_config, training_config = YamlConfigurationParser.parse(CONFIG_FILE_PATH)

    train_dataset = torchvision.datasets.MNIST('./files/', train=True, download=True, transform=Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))]))

    test_dataset = torchvision.datasets.MNIST('./files/', train=False, download=True, transform=Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))]))

    if not on_single_device(devices):
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=training_config.batch_size,
                                               shuffle=False if not on_single_device(devices) else True,
                                               num_workers=multiprocessing.cpu_count(),
                                               sampler=train_sampler if not on_single_device(devices) else None,
                                               pin_memory=torch.cuda.is_available())

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=training_config.batch_size,
                                              shuffle=False if not on_single_device(devices) else True,
                                              num_workers=multiprocessing.cpu_count(),
                                              sampler=valid_sampler if not on_single_device(devices) else None,
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
            .train(training_config.nb_epochs)
