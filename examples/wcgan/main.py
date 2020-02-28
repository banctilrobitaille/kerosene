import logging

from examples.wcgan.models.factories import CustomModelFactory
from examples.wcgan.nn.criterions import CustomCriterionFactory
from examples.wcgan.training.trainers import CycleGanTrainer
from kerosene.configs.configs import RunConfiguration
from kerosene.configs.parsers import YamlConfigurationParser
from kerosene.events.handlers.console import PrintTrainingStatus, PrintMonitors
from kerosene.training.events import Event
from kerosene.training.trainers import ModelTrainerFactory

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    CONFIG_FILE_PATH = "config.yml"

    model_configs, training_config = YamlConfigurationParser.parse(CONFIG_FILE_PATH)

    model_trainers = ModelTrainerFactory(model_factory=CustomModelFactory(),
                                         criterion_factory=CustomCriterionFactory()).create(model_configs)

    train_data_loader = None
    valid_data_loader = None
    test_data_loader = None

    CycleGanTrainer("wcgan", training_config.lambda_A, training_config.lambda_B, training_config.lambda_forward_cycle,
                    training_config.lambda_backward_cycle, train_data_loader, valid_data_loader, test_data_loader,
                    model_trainers, RunConfiguration(use_amp=False)) \
        .with_event_handler(PrintTrainingStatus(every=100), Event.ON_BATCH_END) \
        .with_event_handler(PrintMonitors(every=25), Event.ON_BATCH_END) \
        .train(training_config.nb_epochs)
