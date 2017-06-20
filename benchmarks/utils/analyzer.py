import numpy
import time
import os
import pylab
import matplotlib
import scipy
from circusort import io
from circusort.io.template import TemplateStore
from circusort.io.synthetic import SyntheticStore
from circusort.block.synthetic_generator import Cell
import matplotlib.colors as colors

class Analyzer(object):

    def __init__(self, spk_writer_params, probe, template_store, synthetic_store=None, filtered_data=None, threshold_data=None, start_time=0, stop_time=None):

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
        self.set_cmap_circus('jet')

        if synthetic_store is not None:
            self.synthetic_store = SyntheticStore(os.path.abspath(synthetic_store), 'r')
            self.set_cmap_synthetic('jet')
        else:
            self.synthetic_store = None

        if threshold_data is not None:
            self.threshold_data = numpy.fromfile(threshold_data, dtype=numpy.float32)
            self.threshold_data = self.threshold_data.reshape(self.threshold_data.size/self.nb_channels, self.nb_channels)
        else:
            self.threshold_data = None

        self.start_time = start_time
        if stop_time is None:
            if self.filtered_data is not None:
                self.stop_time = self.filtered_data.shape[0]
            else:
                self.stop_time = None
        else:
            self.stop_time = stop_time

    def set_cmap_synthetic(self, cmap):
        self._cmap      = pylab.get_cmap(cmap)
        self._cNorm     = colors.Normalize(vmin=0, vmax=self.nb_cells)
        self._scalarMap_synthetic = pylab.cm.ScalarMappable(norm=self._cNorm, cmap=self._cmap)

    def set_cmap_circus(self, cmap):
        self._cmap      = pylab.get_cmap(cmap)
        self._cNorm     = colors.Normalize(vmin=0, vmax=self.nb_templates)
        self._scalarMap_circus = pylab.cm.ScalarMappable(norm=self._cNorm, cmap=self._cmap)

    @property
    def nb_channels(self):
        return self.probe.nb_channels

    @property
    def nb_cells(self):
        if self.synthetic_store is not None:
            return self.synthetic_store.nb_cells
        else:
            return 0
    @property
    def nb_templates(self):
        return self.template_store.nb_templates

    def view_positions(self, indices=None, time=None):
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
            all_c += [self._scalarMap_synthetic.to_rgba(int(key))]
        
        pylab.scatter(self.probe.positions[0, :], self.probe.positions[1, :], c='k')
        pylab.scatter(all_x, all_y, c=all_c)
        pylab.show()

    def view_rates(self, indices=None, spacing=1):
        res = self.synthetic_store.get(indices=indices, variables='r')
        pylab.figure()
        for key in res.keys():
            colorVal = self._scalarMap_synthetic.to_rgba(int(key))
            pylab.plot(res[key]['r'] + int(key)*spacing, color=colorVal)
        pylab.xlabel('Time [chunks]')
        pylab.yticks([], [])
        pylab.show()

    def _get_synthetic_template(self, i, time=None, nn=100, hf_dist=45, a_dist=1.0):
        if time is None:
            time = 0
        res  = self.synthetic_store.get(indices=[i], variables=['x', 'y', 'z'])
        cell = Cell(lambda t: res[i]['x'][time], lambda t: res[i]['y'][time], lambda t: res[i]['z'][time], nn=nn, hf_dist=hf_dist, a_dist=a_dist)
        a, b, c = cell.get_waveforms(time, self.probe)
        template = scipy.sparse.csc_matrix((c, (b, a+20)), shape=(self.nb_channels, 81))
        return template

    def view_synthetic_templates(self, indices=None, time=None, nn=100, hf_dist=45, a_dist=1.0):

        if indices is None:
            indices = range(self.nb_cells)

        if not numpy.iterable(indices):
            indices = [indices]

        scaling = None
        pylab.figure()

        for i in indices:

            template   = self._get_synthetic_template(i, time, nn, hf_dist, a_dist)
            template   = template.toarray()
            width      = template.shape[1]
            xmin, xmax = self.probe.field_of_view['x_min'], self.probe.field_of_view['x_max']
            ymin, ymax = self.probe.field_of_view['y_min'], self.probe.field_of_view['y_max']
            if scaling is None:
                scaling= 10*numpy.max(numpy.abs(template))
            colorVal   = self._scalarMap_synthetic.to_rgba(i)
            
            for count, i in enumerate(xrange(self.nb_channels)):
                x, y     = self.probe.positions[:, i]
                xpadding = ((x - xmin)/(float(xmax - xmin) + 1))*(2*width)
                ypadding = ((y - ymin)/(float(ymax - ymin) + 1))*scaling
                pylab.plot(xpadding + numpy.arange(width), ypadding + template[i, :], color=colorVal)
        
        pylab.tight_layout()
        pylab.setp(pylab.gca(), xticks=[], yticks=[])
        pylab.xlim(xmin, 3*width)
        pylab.show()

    def view_circus_templates(self, indices=None):

        if indices is None:
            indices = range(self.nb_templates)

        if not numpy.iterable(indices):
            indices = [indices]

        data      = self.template_store.get(indices, ['templates', 'norms'])
        width     = self.template_store.width
        templates = data.pop('templates').T
        norms     = data.pop('norms')
        scaling   = None
        pylab.figure()

        for count, i in enumerate(indices):

            template   = templates[count].toarray().reshape(self.nb_channels, width) * norms[count]
            xmin, xmax = self.probe.field_of_view['x_min'], self.probe.field_of_view['x_max']
            ymin, ymax = self.probe.field_of_view['y_min'], self.probe.field_of_view['y_max']
            if scaling is None:
                scaling= 10*numpy.max(numpy.abs(template))
            colorVal   = self._scalarMap_circus.to_rgba(i)
            
            for count, i in enumerate(xrange(self.nb_channels)):
                x, y     = self.probe.positions[:, i]
                xpadding = ((x - xmin)/(float(xmax - xmin) + 1))*(2*width)
                ypadding = ((y - ymin)/(float(ymax - ymin) + 1))*scaling
                pylab.plot(xpadding + numpy.arange(width), ypadding + template[i, :], color=colorVal)
        
        pylab.tight_layout()
        pylab.setp(pylab.gca(), xticks=[], yticks=[])
        pylab.xlim(xmin, 3*width)
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
            try:
                spike -= t_min
                tmp1   = all_templates[temp_id].toarray().reshape(self.nb_channels, N_t)
                curve[:, spike-N_t/2:spike+N_t/2+1] += amp*tmp1*norms[temp_id]
            except Exception:
                pass

        pylab.figure()
        for i in xrange(self.nb_channels):
            if self.filtered_data is not None:
                pylab.plot(numpy.arange(t_min, t_max), self.filtered_data[t_min:t_max, i] + i*spacing, '0.5')
            pylab.plot(numpy.arange(t_min, t_max), curve[i, :] + i*spacing, 'r')
        pylab.show()


    def view_thresholds(self, indices=None):
        pylab.figure()
        if indices is None:
            indices = range(self.nb_channels)

        for i in indices:
            pylab.plot(self.threshold_data[:, i], '0.5')

        pylab.plot(numpy.mean(self.threshold_data, 1), 'r')
        pylab.show()

    def compare_rates(self, bin_size=200):

        res   = self.synthetic_store.get(variables=['spike_times'])
        t_max = 0
        for key in res.keys():
            tmp = res[key]['spike_times'].max()
            if tmp > t_max:
                t_max = tmp

        t_max = max(t_max, self.spikes.max()) 
        rates = numpy.zeros((2, int(t_max/bin_size)+1))

        for spike in self.spikes:
            rates[0, int(spike/bin_size)] += 1

        for key in res.keys():
            for spike in res[key]['spike_times']:
                rates[1, int(spike/bin_size)] += 1

        t_appearences = self.template_store.get(variables='times')['times']
        t_appearences = numpy.ceil(t_appearences/bin_size)

        pylab.figure()
        pylab.plot(rates[0])
        pylab.plot(rates[1])
        ymin, ymax = pylab.ylim()
        for t in t_appearences:
            pylab.plot([t, t], [ymin, ymax], 'k--')
        pylab.show()
        return rates

    def get_best_matches(self, indices):
        pass