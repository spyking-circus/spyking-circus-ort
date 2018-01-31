import numpy as np

from circusort.obj.cells import Cells
from circusort.obj.cell import Cell
from circusort.obj.template import Template, TemplateComponent
from circusort.obj.train import Train
from circusort.obj.amplitude import Amplitude


class Spikes(object):
    """The spikes resulting from the sorting.

    Attributes:
        times: numpy.ndarray
            The spike times. An array of shape: (nb_spikes,).
        templates: numpy.ndarray
            The cell (i.e. template) identifiers of each spike. An array of shape: (nb_spikes,).
        amplitudes: numpy.ndarray
            The amplitudes of each spike. An array of shape: (nb_spikes,).
        nb_cells: none | integer (optional)
            The number of cells. The default value is None.
        t_min: none | float (optional)
            The minimum value of the time window of interest. The default value is None.
    """

    def __init__(self, times, templates, amplitudes, nb_cells=None, t_min=None, t_max=None):
        """Initialization.

        Parameters:
            times: numpy.ndarray
                The spike times. An array of shape: (nb_spikes,).
            templates: numpy.ndarray
                The cell (i.e. template) identifiers of each spike. An array of shape: (nb_spikes,).
            amplitudes: numpy.ndarray
                The amplitudes of each spike. An array of shape: (nb_spikes,).
            nb_cells: none | integer (optional)
                The number of cells. The default value is None.
            t_min: none | float (optional)
                The minimum value of the time window of interest. The default value is None.
        """

        assert times.size > 0

        self.times = times
        self.templates = templates
        self.amplitudes = amplitudes
        self.nb_cells = nb_cells
        self.t_min = np.min(self.times) if t_min is None else t_min
        self.t_max = np.max(self.times) if t_max is None else t_max

    def __iter__(self):
        for i in self.cells:
            yield self.get_cell(i)

    def __getitem__(self, index):
        return self.get_cell(index)

    def __len__(self):
        return len(self.cells)

    @property
    def cells(self):
        # TODO add docstring.

        if self.nb_cells is None:
            cells = np.unique(self.templates)
        else:
            if self.nb_cells == np.unique(self.templates).size:
                cells = np.unique(self.templates)
            else:
                cells = np.arange(0, self.nb_cells)

        return cells

    def get_cell(self, k):
        """Get one cell (i.e. cell) given an identifier.

        Parameter:
            k: integer
                The identifier of the cell to get.

        Return:
            cell: circusort.obj.Cell.
                The cell to get.
        """
 
        is_cell = self.templates == k
        times = self.times[is_cell]
        amplitudes = self.amplitudes[is_cell]

        # TODO correct the two following lines (i.e. get the true template).
        channels = np.empty((0,), dtype=np.int)
        waveforms = np.empty((0, 0), dtype=np.float)

        first_component = TemplateComponent(waveforms, channels, 1, amplitudes=[0.8, 1.2])
        template = Template(first_component, 0)
        train = Train(times, t_min=self.t_min, t_max=self.t_max)
        amplitude = Amplitude(amplitudes, times)

        cell = Cell(template, train, amplitude)

        return cell

    def to_cells(self):
        """Convert spikes to cells."""

        cells = {
            k: self.get_cell(k)
            for k in self.cells
        }
        cells = Cells(cells)

        return cells

    def get_time_step(self, selection=None):
        """Get time steps.

        Parameter:
            selection: none | integer | list (optional)
                Unit index or indices. The default value is None.
        Return:
            times: numpy.ndarray
                The spike times to get. An array of shape (nb_spikes,).
        """

        if selection is None:
            times = self.times
        elif isinstance(selection, (int, np.int32)):
            is_selected = np.array([e == selection for e in self.templates])
            times = self.times[is_selected]
        elif isinstance(selection, list):
            is_selected = np.array([e in selection for e in self.templates])
            times = self.times[is_selected]
        else:
            string = "Can't use {} ({}) as a selection."
            message = string.format(selection, type(selection))
            raise NotImplementedError(message)

        return times

    def get_amplitudes(self, selection=None):
        """Get amplitudes.

        Parameter:
            selection: none | integer | list (optional)
                Unit index or indices. The default value is None.
        Return:
            amplitudes: numpy.ndarray
                The amplitudes to get.
        """

        if selection is None:
            amplitudes = self.amplitudes
        elif isinstance(selection, (int, np.int32)):
            is_selected = np.array([e == selection for e in self.templates])
            amplitudes = self.amplitudes[is_selected]
        elif isinstance(selection, list):
            is_selected = np.array([e in selection for e in self.templates])
            amplitudes = self.times[is_selected]
        else:
            string = "Can't use {} ({}) as a selection."
            message = string.format(selection, type(selection))
            raise NotImplementedError(message)

        return amplitudes

    def save(self, path):
        # TODO add docstring.

        raise NotImplementedError()  # TODO complete.
