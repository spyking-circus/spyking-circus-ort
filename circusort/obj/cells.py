import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
import os

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

    def __getitem__(self, identifier):
        # TODO add docstring.

        cell = self.cells[identifier]

        return cell

    def __iter__(self):
        # TODO add docstring.

        iterator = self.cells.itervalues()

        return iterator

    def itervalues(self):
        # TODO add docstring.

        iterator = self.cells.itervalues()

        return iterator

    def iteritems(self):
        # TODO add docstring.

        iterator = self.cells.iteritems()

        return iterator

    @property
    def nb_cells(self):
        # TODO add docstring.

        nb_cells = len(self.cells)

        return nb_cells

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
            self.plot_rates(output=cells_directory, **kwargs)
            self.plot_trains(output=cells_directory, **kwargs)
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
                    path = os.path.join(path, "parameters_rates.pdf")
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
                    path = os.path.join(path, "parameters_trains.pdf")
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
            cell.position.plot(ax=ax, set_ax=False, **kwargs)
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
                    path = os.path.join(path, "parameters_positions.pdf")
                directory = os.path.dirname(path)
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                fig.savefig(path)
        else:
            self._plot_positions(ax, **kwargs)

        return
