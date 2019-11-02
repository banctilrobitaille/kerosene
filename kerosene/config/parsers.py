# -*- coding: utf-8 -*-
# Copyright 2019 Kerosene Authors. All Rights Reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging

import yaml

from kerosene.config.trainers import ModelTrainerConfiguration, TrainerConfiguration
from kerosene.parsers.yaml import CustomYamlParser


class YamlConfigurationParser(object):
    LOGGER = logging.getLogger("YamlConfigurationParser")

    @staticmethod
    def parse(config_file_path):
        with open(config_file_path, 'r') as config_file:
            try:
                config = CustomYamlParser.safe_load(config_file)

                model_trainer_configs = list(
                    map(lambda model_name: ModelTrainerConfiguration.from_dict(model_name,
                                                                               config["models"][model_name]),
                        config["models"]))

                model_trainer_configs = model_trainer_configs if len(model_trainer_configs) > 1 else \
                    model_trainer_configs[0]
                training_config = TrainerConfiguration(config['training'])
                return model_trainer_configs, training_config
            except yaml.YAMLError as e:
                YamlConfigurationParser.LOGGER.warning(
                    "Unable to read the training config file: {} with error {}".format(config_file_path, e))

    @staticmethod
    def parse_section(config_file_path, yml_tag):
        with open(config_file_path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)

                return config[yml_tag]
            except yaml.YAMLError as e:
                YamlConfigurationParser.LOGGER.warning(
                    "Unable to read the training config file: {} with error {}".format(config_file_path, e))
