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
from abc import ABC, abstractmethod

from visdom import Visdom

from kerosene.events.handlers.visdom.data import VisdomData, PlotType


class VisdomPlot(ABC):

    def __init__(self, visdom: Visdom):
        self._visdom = visdom

    @property
    def visdom(self):
        return self._visdom

    @abstractmethod
    def update(self, visdom_data: VisdomData):
        raise NotImplementedError()


class ImagePlot(VisdomPlot):
    def __init__(self, visdom: Visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.image(img=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.image(img=visdom_data.y, win=self._window, **visdom_data.params)


class ImagesPlot(VisdomPlot):
    def __init__(self, visdom: Visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.images(tensor=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.images(tensor=visdom_data.y, win=self._window, **visdom_data.params)


class LinePlot(VisdomPlot):
    def __init__(self, visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.line(X=visdom_data.x, Y=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.line(X=visdom_data.x, Y=visdom_data.y, win=self._window, update='append')


class PiePlot(VisdomPlot):
    def __init__(self, visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.pie(X=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.pie(X=visdom_data.y, win=self._window, **visdom_data.params)


class TextPlot(VisdomPlot):
    def __init__(self, visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.text(visdom_data.y)

        else:
            self._visdom.text(visdom_data.y, win=self._window)


class HistogramPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.histogram(X=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.histogram(X=visdom_data.y, win=self._window, **visdom_data.params)


class ScatterPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.scatter(X=visdom_data.x, Y=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.scatter(X=visdom_data.x, Y=visdom_data.y, win=self._window, **visdom_data.params)


class StemPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.stem(X=visdom_data.x, Y=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.stem(X=visdom_data.x, Y=visdom_data.y, win=self._window, **visdom_data.params)


class HeatmapPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.heatmap(X=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.heatmap(X=visdom_data.y, win=self._window, **visdom_data.params)


class BarPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.bar(X=visdom_data.x, Y=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.bar(X=visdom_data.x, Y=visdom_data.y, win=self._window, **visdom_data.params)


class BoxPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.boxplot(X=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.boxplot(X=visdom_data.x, win=self._window, **visdom_data.params)


class SurfacePlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.surf(X=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.surf(X=visdom_data.y, win=self._window, **visdom_data.params)


class ContourPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.contour(X=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.contour(X=visdom_data.y, win=self._window, **visdom_data.params)


class QuiverPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.quiver(X=visdom_data.x, Y=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.quiver(X=visdom_data.x, Y=visdom_data.y, **visdom_data.params)


class MeshPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.mesh(X=visdom_data.x, Y=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.mesh(X=visdom_data.x, Y=visdom_data.y, **visdom_data.params)


class VisdomPlotFactory(object):

    @staticmethod
    def create_plot(visdom, plot_type):

        if plot_type == PlotType.LINE_PLOT:
            plot = LinePlot(visdom)
        elif plot_type == PlotType.IMAGE_PLOT:
            plot = ImagePlot(visdom)
        elif plot_type == PlotType.IMAGES_PLOT:
            plot = ImagesPlot(visdom)
        elif plot_type == PlotType.PIE_PLOT:
            plot = PiePlot(visdom)
        elif plot_type == PlotType.TEXT_PLOT:
            plot = TextPlot(visdom)
        elif plot_type == PlotType.HISTOGRAM_PLOT:
            plot = HistogramPlot(visdom)
        elif plot_type == PlotType.SCATTER_PLOT:
            plot = ScatterPlot(visdom)
        elif plot_type == PlotType.STEM_PLOT:
            plot = StemPlot(visdom)
        elif plot_type == PlotType.HEATMAP_PLOT:
            plot = HeatmapPlot(visdom)
        elif plot_type == PlotType.BAR_PLOT:
            plot = BarPlot(visdom)
        elif plot_type == PlotType.BOX_PLOT:
            plot = BoxPlot(visdom)
        elif plot_type == PlotType.SURFACE_PLOT:
            plot = SurfacePlot(visdom)
        elif plot_type == PlotType.CONTOUR_PLOT:
            plot = ContourPlot(visdom)
        elif plot_type == PlotType.QUIVER_PLOT:
            plot = QuiverPlot(visdom)
        elif plot_type == PlotType.MESH_PLOT:
            plot = MeshPlot(visdom)
        else:
            raise NotImplementedError("Unable to create a plot for {}".format(plot_type))
        
        return plot
