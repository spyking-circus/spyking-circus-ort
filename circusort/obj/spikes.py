import numpy as np

from circusort.obj.cells import Cells
from circusort.obj.cell import Cell
from circusort.obj.template import Template
from circusort.obj.train import Train
from circusort.obj.amplitude import Amplitude


class Spikes(object):
    # TODO add docstring.

    def __init__(self, times, templates, amplitudes):
        # TODO add docstring.

        self.times = times
        self.templates = templates
        self.amplitudes = amplitudes

    @property
    def unit_identifiers(self):

        # TODO correct the following line (i.e. use the number of units instead).
        unit_identifiers = np.unique(self.templates)

        return unit_identifiers

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
            for k in self.unit_identifiers
        }
        cells = Cells(cells)

        return cells

    def save(self, path):
        # TODO add docstring.

        raise NotImplementedError()  # TODO complete.
