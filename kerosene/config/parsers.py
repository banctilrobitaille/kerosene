import logging

import yaml


class YamlConfigurationParser(object):
    LOGGER = logging.getLogger("YamlConfigurationParser")

    @staticmethod
    def parse(config_file_path):
        with open(config_file_path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)

                return list(map(lambda model_name: ModelTrainingConfiguration(model_name, config["models"][model_name]),
                                config["models"])), TrainingConfiguration(
                    config['training']), VisdomConfiguration.from_dict(config['visdom'])
            except yaml.YAMLError as e:
                logging.warning(
                    "Unable to read the training config file: {} with error {}".format(config_file_path, e))