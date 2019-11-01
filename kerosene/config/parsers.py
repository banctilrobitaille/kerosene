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

from kerosene.parsers.yaml_parser import YamlParser


class YamlConfigurationParser(YamlParser):
    LOGGER = logging.getLogger("YamlConfigurationParser")

    @staticmethod
    def parse_section(config_file_path, yml_tag):
        with open(config_file_path, 'r') as config_file:
            try:
                config = YamlConfigurationParser.load(config_file)

                return config[yml_tag]
            except yaml.YAMLError as e:
                YamlConfigurationParser.LOGGER.warning(
                    "Unable to read the training config file: {} with error {}".format(config_file_path, e))
