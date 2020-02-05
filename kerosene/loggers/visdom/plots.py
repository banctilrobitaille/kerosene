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

from kerosene.loggers.visdom import PlotType
from kerosene.loggers.visdom.data import VisdomData


class VisdomPlot(ABC):

    def __init__(self, visdom: Visdom):
        self._visdom = visdom
        self._window = None

    @property
    def visdom(self):
        return self._visdom

    @property
    def window(self):
        return self._window

    @abstractmethod
    def update(self, visdom_data: VisdomData):
        raise NotImplementedError()


class ImagePlot(VisdomPlot):
    def __init__(self, visdom: Visdom):
        super().__init__(visdom)

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self.visdom.image(img=visdom_data.y, **visdom_data.params)
        else:
            self.visdom.image(img=visdom_data.y, win=self._window, **visdom_data.params)


class ImagesPlot(VisdomPlot):
    def __init__(self, visdom: Visdom):
        super().__init__(visdom)

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.images(tensor=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.images(tensor=visdom_data.y, win=self._window, **visdom_data.params)


class LinePlot(VisdomPlot):
    def __init__(self, visdom):
        super().__init__(visdom)

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.line(X=visdom_data.x, Y=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.line(X=visdom_data.x, Y=visdom_data.y, win=self._window, update='append',
                              name=visdom_data.params['opts']['name'])


class PiePlot(VisdomPlot):
    def __init__(self, visdom):
        super().__init__(visdom)

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.pie(X=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.pie(X=visdom_data.y, win=self._window, **visdom_data.params)


class TextPlot(VisdomPlot):
    def __init__(self, visdom):
        super().__init__(visdom)

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.text(visdom_data.y)

        else:
            self._visdom.text(visdom_data.y, win=self._window)


class HistogramPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.histogram(X=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.histogram(X=visdom_data.y, win=self._window, **visdom_data.params)


class ScatterPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.scatter(X=visdom_data.x, Y=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.scatter(X=visdom_data.x, Y=visdom_data.y, win=self._window, **visdom_data.params)


class StemPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.stem(X=visdom_data.x, Y=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.stem(X=visdom_data.x, Y=visdom_data.y, win=self._window, **visdom_data.params)


class HeatmapPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.heatmap(X=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.heatmap(X=visdom_data.y, win=self._window, **visdom_data.params)


class BoxPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.boxplot(X=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.boxplot(X=visdom_data.x, win=self._window, **visdom_data.params)


class SurfacePlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.surf(X=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.surf(X=visdom_data.y, win=self._window, **visdom_data.params)


class ContourPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.contour(X=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.contour(X=visdom_data.y, win=self._window, **visdom_data.params)


class QuiverPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.quiver(X=visdom_data.x, Y=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.quiver(X=visdom_data.x, Y=visdom_data.y, **visdom_data.params)


class MeshPlot(VisdomPlot):

    def __init__(self, visdom: Visdom):
        super().__init__(visdom)

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.mesh(X=visdom_data.x, Y=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.mesh(X=visdom_data.x, Y=visdom_data.y, **visdom_data.params)


class BarPlot(VisdomPlot):
    def __init__(self, visdom):
        super().__init__(visdom)

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.bar(X=visdom_data.x, Y=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.bar(X=visdom_data.x, Y=visdom_data.y, win=self._window, **visdom_data.params)


class MatplotlibPlot(VisdomPlot):
    def __init__(self, visdom):
        super().__init__(visdom)

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.matplot(plot=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.matplot(plot=visdom_data.y, win=self._window, **visdom_data.params)


class VisdomPlotFactory(object):

    def __init__(self):
        self._plot = {
            PlotType.LINE_PLOT: LinePlot,
            PlotType.IMAGE_PLOT: ImagePlot,
            PlotType.IMAGES_PLOT: ImagesPlot,
            PlotType.PIE_PLOT: PiePlot,
            PlotType.TEXT_PLOT: TextPlot,
            PlotType.BAR_PLOT: BarPlot,
            PlotType.HISTOGRAM_PLOT: HistogramPlot,
            PlotType.SCATTER_PLOT: ScatterPlot,
            PlotType.STEM_PLOT: StemPlot,
            PlotType.HEATMAP_PLOT: HeatmapPlot,
            PlotType.BOX_PLOT: BoxPlot,
            PlotType.SURFACE_PLOT: SurfacePlot,
            PlotType.CONTOUR_PLOT: ContourPlot,
            PlotType.QUIVER_PLOT: QuiverPlot,
            PlotType.MESH_PLOT: MeshPlot,
            PlotType.MATPLOTLIB_PLOT: MatplotlibPlot
        }

    def create_plot(self, visdom, plot_type: PlotType):
        return self._plot[plot_type](visdom)
