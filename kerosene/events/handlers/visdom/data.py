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
from enum import Enum
from typing import Union, List, Optional


class PlotType(Enum):
    LINE_PLOT = "Line Plot"
    IMAGES_PLOT = "Images Plot"
    IMAGE_PLOT = "Image Plot"
    PIE_PLOT = "Pie Plot"
    TEXT_PLOT = "Text Plot"
    HISTOGRAM_PLOT = "Histogram Plot"
    SCATTER_PLOT = "Scatter Plot"
    STEM_PLOT = "Stem Plot"
    HEATMAP_PLOT = "Heatmap Plot"
    BAR_PLOT = "Bar Plot"
    BOX_PLOT = "Box Plot"
    SURFACE_PLOT = "Surface Plot"
    CONTOUR_PLOT = "Contour Plot"
    QUIVER_PLOT = "Quiver Plot"
    MESH_PLOT = "Mesh Plot"

    def __str__(self):
        return self.value


class PlotFrequency(Enum):
    EVERY_STEP = "Step"
    EVERY_EPOCH = "Epoch"

    def __str__(self):
        return self.value


class VisdomData(object):

    def __init__(self, source_name, variable_name: Union[List[str], str], plot_type: PlotType,
                 plot_frequency: Optional[PlotFrequency], x, y, params: dict = None):
        self._source_name = source_name
        self._variable_name = variable_name
        self._plot_type = plot_type
        self._plot_frequency = plot_frequency
        self._x = x
        self._y = y
        self._params = params

    @property
    def id(self):
        return hash(self)

    @property
    def source_name(self):
        return self._source_name

    @property
    def variable_name(self):
        return self._variable_name

    @property
    def plot_type(self):
        return self._plot_type

    @property
    def plot_frequency(self):
        return self._plot_frequency

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def params(self):
        return self._params

    def __hash__(self):
        return hash(self._source_name + self._variable_name + str(self._plot_frequency) + str(self._plot_type))
