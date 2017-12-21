import numpy as np

from circusort.obj.cells import Cells
from circusort.obj.cell import Cell
from circusort.obj.template import Template
from circusort.obj.train import Train
from circusort.obj.amplitude import Amplitude


class Spikes(object):
    """The spikes resulting from the sorting.

    Attributes:
        times: numpy.ndarray
            The spike times. An array of shape: (nb_spikes,).
        templates: numpy.ndarray
            The unit (i.e. template) identifiers of each spike. An array of shape: (nb_spikes,).
        amplitudes: numpy.ndarray
            The amplitudes of each spike. An array of shape: (nb_spikes,).
    """

    def __init__(self, times, templates, amplitudes):
        """Initialization.

        Parameters:
            times: numpy.ndarray
                The spike times. An array of shape: (nb_spikes,).
            templates: numpy.ndarray
                The unit (i.e. template) identifiers of each spike. An array of shape: (nb_spikes,).
            amplitudes: numpy.ndarray
                The amplitudes of each spike. An array of shape: (nb_spikes,).
        """

        self.times = times
        self.templates = templates
        self.amplitudes = amplitudes

    @property
    def units(self):
        # TODO add docstring.

        # TODO correct the following line (i.e. use the number of units instead).
        units = np.unique(self.templates)

        return units

    @property
    def nb_units(self):
        # TODO add docstring.

        nb_units = self.units.size

        return nb_units

    def get_unit(self, k):
        """Get one unit (i.e. cell) given an identifier.

        Parameter:
            k: integer
                The identifier of the unit to get.

        Return:
            unit: circusort.obj.Cell.
                The unit to get.
        """

        is_unit = self.templates == k
        times = self.times[is_unit]
        amplitudes = self.amplitudes[is_unit]

        # TODO correct the two following lines (i.e. get the true template).
        channels = np.empty((0,), dtype=np.int)
        waveforms = np.empty((0, 0), dtype=np.float)

        template = Template(channels, waveforms)
        train = Train(times)
        amplitude = Amplitude(amplitudes, times)

        unit = Cell(template, train, amplitude)

        return unit

    def to_units(self):
        """Convert spikes to cells."""

        cells = {
            k: self.get_unit(k)
            for k in self.units
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
