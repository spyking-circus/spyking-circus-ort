import numpy
import time
import os
import pylab
import matplotlib
from circusort import io
from circusort.io.template import TemplateStore
from circusort.io.synthetic import SyntheticStore
import matplotlib.colors as colors

class Analyzer(object):

    def __init__(self, spk_writer_params, probe, template_store, synthetic_store=None, filtered_data=None):

        self.probe    = io.Probe(probe)

        self.spikes   = numpy.fromfile(spk_writer_params['spike_times'], dtype=numpy.int32)
        self.temp_ids = numpy.fromfile(spk_writer_params['templates'], dtype=numpy.int32)
        self.amps     = numpy.fromfile(spk_writer_params['amplitudes'], dtype=numpy.float32)

        if filtered_data is not None:
            self.filtered_data = numpy.fromfile(filtered_data, dtype=numpy.float32)
            self.filtered_data = self.filtered_data.reshape(self.filtered_data.size/self.nb_channels, self.nb_channels)
        else:
            self.filtered_data = None

        self.template_store = TemplateStore(os.path.join(os.path.abspath(template_store), 'template_store.h5'), 'r')

        if synthetic_store is not None:
            self.synthetic_store = SyntheticStore(os.path.abspath(synthetic_store), 'r')
            self.set_cmap('jet')


    def set_cmap(self, cmap):
        self._cmap      = pylab.get_cmap(cmap)
        self._cNorm     = colors.Normalize(vmin=0, vmax=self.nb_cells)
        self._scalarMap = pylab.cm.ScalarMappable(norm=self._cNorm, cmap=self._cmap)

    @property
    def nb_channels(self):
        return self.probe.nb_channels

    @property
    def nb_cells(self):
        return self.synthetic_store.nb_cells

    def show_positions(self, indices=None, time=None):
        if time is None:
            time = 0
        res = self.synthetic_store.get(indices=indices, variables=['x', 'y', 'z'])
        pylab.figure()

        all_x = []
        all_y = []
        all_z = []
        all_c = []

        for key in res.keys():
            all_x += [res[key]['x'][time]]
            all_y += [res[key]['y'][time]]
            all_z += [res[key]['z'][time]]
            all_c += [self._scalarMap.to_rgba(int(key))]
        
        pylab.scatter(self.probe.positions[0, :], self.probe.positions[1, :], c='k')
        pylab.scatter(all_x, all_y, c=all_c)
        pylab.show()

    def show_rates(self, indices=None, spacing=1):
        res = self.synthetic_store.get(indices=indices, variables='r')
        pylab.figure()
        for key in res.keys():
            colorVal = self._scalarMap.to_rgba(int(key))
            pylab.plot(res[key]['r'] + int(key)*spacing, color=colorVal)
        pylab.xlabel('Time [chunks]')
        pylab.yticks([], [])
        pylab.show()


    def view_time_slice(self, t_min=None, t_max=None, spacing=10):

        nb_buffers = 10
        nb_samples = 1024

        if t_max is None:
            t_max = self.spikes.max() + nb_samples

        if t_min is None:
            t_min = t_max - nb_buffers * nb_samples

        N_t           = self.template_store.width
        data          = self.template_store.get()
        all_templates = data.pop('templates').T
        norms         = data.pop('norms')
        curve         = numpy.zeros((self.nb_channels, t_max-t_min), dtype=numpy.float32)
        idx           = numpy.where(self.spikes > t_min)[0]

        for spike, temp_id, amp in zip(self.spikes[idx], self.temp_ids[idx], self.amps[idx]):
            if spike > t_min + N_t/2:
                spike -= t_min
                tmp1   = all_templates[temp_id].toarray().reshape(self.nb_channels, N_t)
                curve[:, spike-N_t/2:spike+N_t/2+1] += amp*tmp1*norms[temp_id]
            
        pylab.figure()
        for i in xrange(self.nb_channels):
            pylab.plot(numpy.arange(t_min, t_max), self.filtered_data[t_min:t_max, i] + i*spacing, '0.5')
            pylab.plot(numpy.arange(t_min, t_max), curve[i, :] + i*spacing, 'r')
        pylab.show()