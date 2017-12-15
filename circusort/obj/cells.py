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
            cells: dictionary
                Dictionary of cells to save.
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
            for k, cell in self.iteritems():
                cell_directory = os.path.join(cells_directory, "{}".format(k))
                cell.plot(output=cell_directory, **kwargs)

        return
