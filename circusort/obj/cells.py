from circusort.io.parameter.cells import get_cells_parameters


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
