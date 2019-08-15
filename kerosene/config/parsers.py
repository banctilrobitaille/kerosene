import logging

import yaml

from kerosene.config.trainers import ModelTrainerConfiguration, TrainingConfiguration


class YamlConfigurationParser(object):
    LOGGER = logging.getLogger("YamlConfigurationParser")

    @staticmethod
    def parse(config_file_path):
        with open(config_file_path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)

                model_trainer_configs = list(
                    map(lambda model_name: ModelTrainerConfiguration.from_dict(model_name,
                                                                               config["models"][model_name]),
                        config["models"]))
                training_config = TrainingConfiguration(config['training'])
                return model_trainer_configs, training_config
            except yaml.YAMLError as e:
                YamlConfigurationParser.LOGGER.warning(
                    "Unable to read the training config file: {} with error {}".format(config_file_path, e))
