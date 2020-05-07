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
            The number of cells.
            The default value is None.
        t_min: none | float (optional)
            The minimum value of the time window of interest.
            The default value is None.
        t_max: none | float (optional)
            The maximum value of the time window of interest.
            The default value is None.
    """

    def __init__(self, times, templates, amplitudes, nb_cells=None, t_min=None, t_max=None):
        """Initialization.

        Arguments:
            times: numpy.ndarray
                The spike times. An array of shape: (nb_spikes,).
            templates: numpy.ndarray
                The cell (i.e. template) identifiers of each spike. An array of shape: (nb_spikes,).
            amplitudes: numpy.ndarray
                The amplitudes of each spike. An array of shape: (nb_spikes,).
            nb_cells: none | integer (optional)
                The number of cells.
                The default value is None.
            t_min: none | float (optional)
                The minimum value of the time window of interest.
                The default value is None.
            t_max: none | float (optional)
                The maximum value of the time window of interest.
                The default value is None.
        """

        # assert times.size > 0

        self.times = times
        self.templates = templates
        self.amplitudes = amplitudes
        self.nb_cells = nb_cells
        self.t_min = (np.min(self.times) if self.times.size > 0 else None) if t_min is None else t_min
        self.t_max = (np.max(self.times) if self.times.size > 0 else None) if t_max is None else t_max

    def __iter__(self):

        for i in self.ids:
            yield self.get_cell(i)

    def __getitem__(self, index):

        return self.get_cell(index)

    def __len__(self):

        return len(self.ids)

    @property
    def ids(self):

        if self.nb_cells is None:
            ids = np.unique(self.templates)
        else:
            if self.nb_cells == np.unique(self.templates).size:
                ids = np.unique(self.templates)
            else:
                ids = np.arange(0, self.nb_cells)

        return ids

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
        amplitude = Amplitude(amplitudes, times, t_min=self.t_min, t_max=self.t_max)

        cell = Cell(template, train, amplitude)

        return cell

    def to_cells(self):
        """Convert spikes to cells."""

        cells = {
            k: self.get_cell(k)
            for k in self.ids
        }
        
        cells = Cells(cells)

        return cells

    def get_spike_data(self, t_min=None, t_max=None, indices=None):

        results = {'spike_times' : [], 'templates' : [], 'amplitudes' : []}
        if indices is None:
            indices = range(len(self))

        for i in indices:
            spikes = self.get_cell(i).train.times
            amplitudes = self.get_cell(i).amplitude.amplitudes
            if t_min is not None:
                mask = spikes >= t_min
                spikes = spikes[mask]
                amplitudes = amplitudes[mask]
            if t_max is not None:
                mask = spikes < t_max
                spikes = spikes[mask]
                amplitudes = amplitudes[mask]
            templates = i*np.ones(len(spikes), dtype=np.int32)
            results['spike_times'] += [spikes]
            results['templates'] += [templates]
            results['amplitudes'] += [amplitudes]

        if len(indices) > 0:
            for key in ['spike_times', 'amplitudes', 'templates']:
                results[key] = np.concatenate(results[key])

        return results

    def save(self, path):

        raise NotImplementedError()  # TODO complete.
