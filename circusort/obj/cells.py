import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
import os
import numpy as np

from collections import OrderedDict

from circusort.obj.matches import Matches
from circusort.obj.similarities import Similarities
from circusort.io.parameter.cells import get_cells_parameters
from circusort.utils.path import normalize_path


class Cells(object):
    """Cell model.

    Attributes:
        cells: dictionary
            The cells.
    """

    def __init__(self, cells, parameters=None):
        """Initialization.

        Parameters:
            cells: dictionary
                The cells.
            parameters: circusort.obj.CellsParameters
                The parameters of the cells.
        """

        self.cells = cells

        self.parameters = get_cells_parameters() if parameters is None else parameters

        self._mode = None

    def __len__(self):

        return len(self.cells)

    @property
    def ids(self):

        return list(self.cells.keys())

    def __getitem__(self, identifier):

        cell = self.cells[identifier]

        return cell

    def __iter__(self):

        iterator = iter(self.values())

        return iterator

    def keys(self):

        iterator = self.cells.keys()

        return iterator

    def values(self):

        iterator = self.cells.values()

        return iterator

    def items(self):

        iterator = self.cells.items()

        return iterator

    @property
    def nb_cells(self):

        nb_cells = len(self.cells)

        return nb_cells

    def slice_by_ids(self, indices):
        
        cells = OrderedDict()
        for key in indices:
            if key in self.ids:
                cells[key] = self.cells[key]
        
        return Cells(cells)

    def slice_by_time(self, t_min=None, t_max=None):

        cells = OrderedDict()
        for key, value in self.items():
            cells[key] = value.slice(t_min, t_max)

        return Cells(cells)

    @property
    def t_min(self):
        
        t_min = np.inf 
        for c in self:
            if c.train.t_min < t_min:
                t_min = c.train.t_min
        
        return t_min

    @property
    def t_max(self):
        
        t_max = 0 
        for c in self:
            if c.train.t_max > t_max:
                t_max = c.train.t_max
        
        return t_max

    def set_t_min(self, t_min):

        for c in self:
            c.train = c.train.slice(t_min, self.t_max)

        return

    def set_t_max(self, t_max):

        for c in self:
            c.train = c.train.slice(self.t_min, t_max)

        return

    def save(self, path, mode='default', **kwargs):
        """Save cells to files.

        Parameters:
            path: string
                The path to the directory in which to save the cells.
            mode: string (optional)
                The mode to use to save the cells. Either 'default', 'by cells' or 'by components'. The default value is
                'default'.
        """

        if mode == 'default' or mode == 'by cells':

            path = normalize_path(path, **kwargs)
            cells_directory = os.path.join(path, "cells")
            self.parameters.save(cells_directory, **kwargs)
            for k, cell in self.items():
                cell_directory = os.path.join(cells_directory, "{}".format(k))
                cell.save(cell_directory)

            # Update private attributes.
            self._mode = 'default'

        elif mode == 'by components':

            raise NotImplementedError()  # TODO complete.

        else:

            message = "Unknown mode value: {}".format(mode)
            raise ValueError(message)

        return

    @property
    def mean_rate(self):

        mean_rate = np.mean([c.mean_rate for c in self])

        return mean_rate

    @property
    def trains(self):

        trains = np.array([
            cell.train
            for cell in self
        ])

        return trains

    def rate(self, time_bin=1):

        bins = np.arange(self.t_min, self.t_max, time_bin)
        result = np.zeros((len(self), len(bins) - 1))

        for count, c in enumerate(self):
            result[count, :] = c.rate(time_bin)

        return result

    def plot(self, output=None, **kwargs):

        if output is None:
            raise NotImplementedError()  # TODO complete.
        else:
            path = normalize_path(output, **kwargs)
            cells_directory = os.path.join(path, "cells")
            parameters = get_cells_parameters(cells_directory)
            kwargs.update(parameters['general'])
            try:
                self.plot_rates(output=cells_directory, **kwargs)
            except NotImplementedError:
                pass  # TODO remove try ... except ...
            self.plot_trains(output=cells_directory, **kwargs)
            # TODO plot amplitudes.
            self.plot_positions(output=cells_directory, **kwargs)
            for k, cell in self.items():
                cell_directory = os.path.join(cells_directory, "{}".format(k))
                cell.plot(output=cell_directory, **kwargs)

        return

    def _plot_rates(self, ax, **kwargs):

        for k, cell in self.items():
            cell.plot_rate(ax=ax, **kwargs)
        ax.set_title(u"Rates")

        return

    def plot_rates(self, output=None, ax=None, **kwargs):

        if output is not None and ax is None:
            plt.ioff()

        if ax is None:
            fig = plt.figure()
            gs = gds.GridSpec(1, 1)
            ax_ = fig.add_subplot(gs[0])
            self._plot_rates(ax_, **kwargs)
            gs.tight_layout(fig)
            if output is None:
                fig.show()
            else:
                path = normalize_path(output)
                if path[-4:] != ".pdf":
                    path = os.path.join(path, "rates.pdf")
                directory = os.path.dirname(path)
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                fig.savefig(path)
        else:
            self._plot_rates(ax, **kwargs)

        return

    def _plot_trains(self, ax, **kwargs):

        for k, cell in self.items():
            cell.train.plot(ax=ax, offset=k, **kwargs)
        ax.set_yticks([i for i in range(0, self.nb_cells)])
        ax.set_yticklabels([str(k) for k in range(0, self.nb_cells)])
        ax.set_title(u"Trains")

        return

    def plot_trains(self, output=None, ax=None, **kwargs):

        if output is not None and ax is None:
            plt.ioff()

        if ax is None:
            fig = plt.figure()
            gs = gds.GridSpec(1, 1)
            ax_ = fig.add_subplot(gs[0])
            self._plot_trains(ax_, **kwargs)
            gs.tight_layout(fig)
            if output is None:
                fig.show()
            else:
                path = normalize_path(output)
                if path[-4:] != ".pdf":
                    path = os.path.join(path, "trains.pdf")
                directory = os.path.dirname(path)
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                fig.savefig(path)
        else:
            self._plot_trains(ax, **kwargs)

        return

    def _plot_positions(self, ax, probe=None, **kwargs):

        if probe is not None:
            probe.plot(ax=ax, **kwargs)
        for k, cell in self.items():
            if cell.position is not None:  # TODO be able to remove this line.
                color = 'C{}'.format(1 + k % 9)
                cell.position.plot(ax=ax, color=color, set_ax=False, **kwargs)
        ax.set_title(u"Positions")

        return

    def plot_positions(self, output=None, ax=None, **kwargs):

        if output is not None and ax is None:
            plt.ioff()

        if ax is None:
            fig = plt.figure()
            gs = gds.GridSpec(1, 1)
            ax_ = fig.add_subplot(gs[0])
            self._plot_positions(ax_, **kwargs)
            gs.tight_layout(fig)
            if output is None:
                fig.show()
            else:
                path = normalize_path(output)
                if path[-4:] != ".pdf":
                    path = os.path.join(path, "positions.pdf")
                directory = os.path.dirname(path)
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                fig.savefig(path)
        else:
            self._plot_positions(ax, **kwargs)

        return

    def get_parameters(self):
        """Get the parameters of the cells.

        Return:
            parameters: dictionary
                A dictionary which contains the parameters of the cells.
        """

        parameters = {
            'mode': self._mode
        }
        # TODO correct.

        return parameters

    def compute_similarities(self, cells):
        """Compute the similarities between two set of cells.

        Argument:
            cells: circusort.obj.Cells
                The set of cells with which similarities have to be computed.
        Return:
            similarities: numpy.ndarray
                The matrix of similarities between the two set of cells.
        """

        similarities = Similarities(self, cells)

        return similarities

    def compute_matches(self, cells, threshold=0.9, t_min=None, t_max=None):
        """Compute the matches between two set of cells.

        Attribute:
            cells: circusort.obj.Cells
                The set of cells with which matches have to be computed.
            threshold: float (optional)
                The similarity threshold to use to detect potential matches.
                The default value is 0.9.
            t_min: none | float (optional)
                The start time of the window to use to compare trains.
                The default value is None.
            t_max: none | float (optional)
                The end time of the window to use to compare trains.
                The default value is None.
        Return:
            matches: tuple
                The matches between the two set of cells.
        """

        matches = Matches(self, cells, threshold=threshold, t_min=t_min, t_max=t_max)

        return matches
