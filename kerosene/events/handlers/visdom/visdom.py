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
from typing import List, Union, Dict

from visdom import Visdom

from kerosene.events.handlers.base_handler import EventHandler
from kerosene.events.handlers.visdom.config import VisdomConfiguration
from kerosene.events.handlers.visdom.data import VisdomData
from kerosene.events.handlers.visdom.plots import VisdomPlot, VisdomPlotFactory


class VisdomLogger(EventHandler):
    def __init__(self, visdom_config: VisdomConfiguration):
        self._visdom_config = visdom_config
        self._visdom = Visdom(server=visdom_config.server, port=visdom_config.port, env=visdom_config.env)
        self._plots: Dict[int, VisdomPlot] = {}

    @property
    def visdom_config(self):
        return self._visdom_config

    @property
    def plots(self):
        return self._plots

    def __call__(self, visdom_data: Union[List[VisdomData], VisdomData]):
        visdom_data = [visdom_data] if not hasattr(visdom_data, '__iter__') else visdom_data

        for visdom_datum in visdom_data:
            if visdom_datum.id not in self._plots.keys():
                self._plots[visdom_datum.id] = VisdomPlotFactory.create_plot(self._visdom, visdom_datum.plot_type)

            self._plots[visdom_datum.id].update(visdom_datum)
