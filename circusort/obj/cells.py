import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
import os
import numpy as np

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

        return self.cells.keys()

    def __getitem__(self, identifier):
        # TODO add docstring.

        cell = self.cells[identifier]

        return cell

    def __iter__(self):
        # TODO add docstring.

        iterator = self.values()

        return iterator

    def keys(self):
        # TODO add docstring.

        iterator = self.cells.iterkeys()

        return iterator

    def values(self):
        # TODO add docstring.

        iterator = self.cells.itervalues()

        return iterator

    def items(self):
        # TODO add docstring.

        iterator = self.cells.iteritems()

        return iterator

    @property
    def nb_cells(self):
        # TODO add docstring.

        nb_cells = len(self.cells)

        return nb_cells

    def slice_by_ids(self, indices):
        
        cells = {}
        for key in indices:
            if key in self.ids:
                cells[key] = self.cells[key]
        
        return Cells(cells)

    def slice_by_time(self, t_min=None, t_max=None):

        cells = {}
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
            for k, cell in self.iteritems():
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

    def plot(self, output=None, **kwargs):
        # TODO add docstring.

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
            for k, cell in self.iteritems():
                cell_directory = os.path.join(cells_directory, "{}".format(k))
                cell.plot(output=cell_directory, **kwargs)

        return

    def _plot_rates(self, ax, **kwargs):

        for k, cell in self.iteritems():
            cell.plot_rate(ax=ax, **kwargs)
        ax.set_title(u"Rates")

        return

    def plot_rates(self, output=None, ax=None, **kwargs):
        # TODO add docstring.

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

        for k, cell in self.iteritems():
            cell.train.plot(ax=ax, offset=k, **kwargs)
        ax.set_yticks([i for i in range(0, self.nb_cells)])
        ax.set_yticklabels([str(k) for k in range(0, self.nb_cells)])
        ax.set_title(u"Trains")

        return

    def plot_trains(self, output=None, ax=None, **kwargs):
        # TODO add docstring.

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
        # TODO add docstring.

        if probe is not None:
            probe.plot(ax=ax, **kwargs)
        for k, cell in self.iteritems():
            if cell.position is not None:  # TODO be able to remove this line.
                color = 'C{}'.format(1 + k % 9)
                cell.position.plot(ax=ax, color=color, set_ax=False, **kwargs)
        ax.set_title(u"Positions")

        return

    def plot_positions(self, output=None, ax=None, **kwargs):
        # TODO add docstring.

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
