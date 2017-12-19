from circusort.obj.cells import Cells
from circusort.obj.cell import Cell
from circusort.obj.train import Train
from circusort.obj.amplitude import Amplitude


class Spikes(object):
    # TODO add docstring.

    def __init__(self, times, templates, amplitudes):
        # TODO add docstring.

        self.times = times
        self.templates = templates
        self.amplitudes = amplitudes

    def get_unit(self, k):
        # TODO add docstring.

        is_unit = self.templates == k
        times = self.times[is_unit]
        amplitudes = self.amplitudes[is_unit]

        template = None  # TODO correct.
        train = Train(times)
        amplitude = Amplitude(amplitudes, times)

        unit = Cell(template, train, amplitude)

        return unit

    def to_units(self):
        # TODO add docstring.

        cells = {}
        cells = Cells(cells)

        return cells

    def save(self, path):
        # TODO add docstring.

        raise NotImplementedError()  # TODO complete.