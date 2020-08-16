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
from kerosene.configs.parsers import YamlConfigurationParser


class VisdomConfiguration(object):
    def __init__(self, port, server, env, filename, offline=False):
        self._port = port
        self._server = server
        self._env = env
        self._filename = filename
        self._offline = offline

    @property
    def port(self):
        return self._port

    @property
    def server(self):
        return self._server

    @property
    def env(self):
        return self._env

    @property
    def filename(self):
        return self._filename

    @property
    def offline(self):
        return self._offline

    @classmethod
    def from_dict(cls, config_dict):
        return cls(config_dict.get("port", 8097), config_dict.get("server", "http://localhost"),
                   config_dict.get("env", "main"), config_dict.get("filename", None),
                   config_dict.get("offline", False))

    @classmethod
    def from_yml(cls, yml_file, yml_tag="visdom"):
        config = YamlConfigurationParser.parse_section(yml_file, yml_tag)
        return VisdomConfiguration.from_dict(config)
