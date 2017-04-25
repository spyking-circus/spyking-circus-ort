from .block import Block
import numpy
import time
import scipy.interpolate
from circusort.config.probe import Probe
from circusort.utils.algorithms import PCAEstimator
from circusort.utils.clustering import rho_estimation, density_based_clustering
from circusort.utils.buffer import DictionaryBuffer

class Density_clustering(Block):
    '''TODO add docstring'''

    name = "Density Clustering"

    params = {'alignment'     : True,
              'time_constant' : 30.,
              'sampling_rate' : 20000.,
              'spike_width'   : 5,
              'nb_waveforms'  : 10000, 
              'probe'         : None,
              'radius'        : None,
              'm_ratio'       : 0.01,
              'noise_thr'     : 0.8,
              'n_min'         : 0.002,
              'dispersion'    : [5, 5],
              'extraction'    : 'median-raw', 
              'two_components': True}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        if self.probe == None:
            self.log.error('{n}: the probe file must be specified!'.format(n=self.name))
        else:
            self.probe = Probe(self.probe, radius=self.radius, logger=self.log)
            self.log.info('{n} reads the probe layout'.format(n=self.name))
        self.add_input('data')
        self.add_input('pcs')
        self.add_input('peaks')
        self.add_input('mads')
        self.add_output('templates', 'dict')

    def _initialize(self):

        self._spike_width_ = int(self.sampling_rate*self.spike_width*1e-3)
        self.sign_peaks    = []
        self.receive_pcs   = True
        if numpy.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width = (self._spike_width_-1)//2

        self.buffer = DictionaryBuffer()

        if self.alignment:
            self.cdata = numpy.linspace(-self._width, self._width, 5*self._spike_width_)
            self.xdata = numpy.arange(-2*self._width, 2*self._width + 1)
        return

    @property
    def nb_channels(self):
        return self.inputs['data'].shape[0]

    @property
    def nb_samples(self):
        return self.inputs['data'].shape[1]

    def _get_all_valid_peaks(self, peaks, shuffle=True):
        all_peaks = {}
        for key in peaks.keys():
            all_peaks[key] = set([])
            for channel in peaks[key].keys():
                 all_peaks[key] = all_peaks[key].union(peaks[key][channel])

            all_peaks[key] = numpy.array(list(all_peaks[key]), dtype=numpy.int32)
            mask           = self._is_valid(all_peaks[key])
            all_peaks[key] = all_peaks[key][mask]
            if shuffle:
                all_peaks[key] = numpy.random.permutation(all_peaks[key])
        return all_peaks

    def _remove_nn_peaks(self, peak, peaks):
        mask = numpy.abs(peaks - peak) > self._width
        return peaks[mask]

    def _is_valid(self, peak):
        if self.alignment:
            return (peak >= 2*self._width) & (peak + 2*self._width < self.nb_samples)
        else:
            return (peak >= self._width) & (peak + self._width < self.nb_samples)

    def _get_extrema_indices(self, peak, peaks):
        res = []
        for key in peaks.keys():
            for channel in peaks[key].keys():
                if peak in peaks[key][channel]:
                    res += [int(channel)]
        return res

    def _get_best_channel(self, batch, key, peak, peaks):

        indices = self._get_extrema_indices(peak, peaks)

        if key == 'negative':
            channel = numpy.argmin(batch[indices, peak])
            is_neg  = True
        elif key == 'positive':
            channel = numpy.argmax(batch[indices, peak])
            is_neg  = False
        elif key == 'both':
            if numpy.abs(numpy.max(batch[indices, peak])) > numpy.abs(numpy.min(batch[indices, peak])):
                channel = numpy.argmax(batch[indices, peak])
                is_neg  = False
            else:
                channel = numpy.argmin(batch[indices, peak])
                is_neg = True

        return indices[channel], is_neg

    def _get_snippet(self, batch, channel, peak, is_neg):
        indices = self.probe.edges[channel]
        if self.alignment:    
            idx     = self.chan_positions[channel]
            zdata   = batch[indices, peak - 2*self._width:peak + 2*self._width + 1]
            ydata   = numpy.arange(len(indices))

            if len(ydata) == 1:
                f        = scipy.interpolate.UnivariateSpline(self.xdata, zdata, s=0)
                if is_neg:
                    rmin = (numpy.argmin(f(self.cdata)) - len(self.cdata)/2.)/5.
                else:
                    rmin = (numpy.argmax(f(self.cdata)) - len(self.cdata)/2.)/5.
                ddata    = numpy.linspace(rmin - self._width, rmin + self._width, self._spike_width_)
                sub_mat  = f(ddata).astype(numpy.float32).reshape(1, self._spike_width_)
            else:
                f        = scipy.interpolate.RectBivariateSpline(ydata, self.xdata, zdata, s=0, kx=min(len(ydata)-1, 3))
                if is_neg:
                    rmin = (numpy.argmin(f(idx, self.cdata)[0, :]) - len(self.cdata)/2.)/5.
                else:
                    rmin = (numpy.argmax(f(idx, self.cdata)[0, :]) - len(self.cdata)/2.)/5.
                ddata    = numpy.linspace(rmin-self._width, rmin+self._width, self._spike_width_)
                sub_mat  = f(ydata, ddata).astype(numpy.float32)
        else:
            sub_mat = batch[indices, peak - self._width:peak + self._width + 1]

        return sub_mat

    def _guess_output_endpoints(self):
        if self.inputs['data'].dtype is not None:
            self.decay_time = numpy.exp(-self.nb_samples/float(self.time_constant))
            self.chan_positions = numpy.zeros(self.nb_channels, dtype=numpy.int32)
            self.sub_pcas = {}
            for channel in xrange(self.nb_channels):
                self.chan_positions[channel] = numpy.where(self.probe.edges[channel] == channel)[0]

    def _get_sub_pca(self, channel):
        if self.sub_pcas.has_key(channel):
            return self.sub_pcas[channel]
        else:
            pass

    def _init_data_structures(self):
        self.pca_data   = {}
        self.raw_data   = {}
        self.clusters   = {}
        self.templates  = {}

        self.templates['dat'] = {}
        self.templates['amp'] = {}
        if self.two_components:
            self.templates['two'] = {}

        if not numpy.all(self.pcs[0] == 0):
            self.sign_peaks += ['negative']
        if not numpy.all(self.pcs[1] == 0):
            self.sign_peaks += ['positive']
        self.log.debug("{n} will detect peaks {s}".format(n=self.name, s=self.sign_peaks))

        for key in self.sign_peaks:
            self.pca_data[key]   = {}
            self.raw_data[key]   = {}
            self.clusters[key]   = {}
            self.templates['dat'][key] = {}
            self.templates['amp'][key] = {}
            if self.two_components:
                self.templates['two'][key] = {}

        for key in self.sign_peaks:
            for channel in xrange(self.nb_channels):
                self._reset_data_structures(key, channel)


    def _perform_clustering(self, key, channel):
        a, b, c = self.pca_data[key][channel].shape
        self.log.debug("{n} clusters {m} {k} waveforms on channel {d}".format(n=self.name_and_counter, m=a, k=key, d=channel))
        data    = self.pca_data[key][channel].reshape(a, b*c)
        n_min   = numpy.maximum(20, int(self.n_min*a))
        rho, dist, nb_selec = rho_estimation(data, mratio=self.m_ratio)
        self.clusters[key][channel], c = density_based_clustering(rho, dist, smart_select=True, n_min=n_min)
        ### SHould we add the merging step
        self._update_templates(key, channel)


    def _update_templates(self, key, channel):

        labels = numpy.unique(self.clusters[key][channel])
        labels = labels[labels > -1]

        if len(labels) > 0:
            self.log.debug("{n} found {m} templates on channel {d}".format(n=self.name_and_counter, m=len(labels), d=channel))
        for l in labels:
            indices = numpy.where(self.clusters[key][channel] == l)[0]
            data = self.raw_data[key][channel][indices]
            if self.extraction == 'mean-raw':
                template = numpy.mean(data, 0)
            elif self.extraction == 'median-raw':
                template = numpy.median(data, 0)

            template   = template.T
            template   = self._center_template(template, key)

            
            amplitudes = self._get_amplitudes(data, template, channel)
            
            # import pylab
            # pylab.figure()
            # for i in xrange(len(template)):
            #     if i == self.chan_positions[channel]:
            #         c = 'r'
            #     else: 
            #         c = '0.5'
            #     pylab.plot(template[i, :], c=c)
            # xmin, xmax = pylab.xlim()
            # pylab.plot([xmin, xmax], [-self.thresholds[channel], -self.thresholds[channel]], 'k--')
            # pylab.title("nb_samples %d" %len(indices))
            # pylab.savefig("test_%d_%s_%d_%d.png" %(self.counter, key, channel, l))

            self.templates['dat'][key][channel] = numpy.vstack((self.templates['dat'][key][channel], template.reshape(1, template.shape[0], template.shape[1])))
            self.templates['amp'][key][channel] = numpy.vstack((self.templates['amp'][key][channel], amplitudes))

            if self.two_components:
                self.templates['two'][key][channel] = numpy.vstack((self.templates['two'][key][channel], template2.reshape(1, template2.shape[0], template2.shape[1])))

        self.to_reset += [(key, channel)]

    def _get_amplitudes(self, data, template, channeln template2=None):
        x, y, z     = data.shape
        data        = data.reshape(x, y*z)
        first_flat  = template.reshape(y*z, 1)
        amplitudes  = numpy.dot(data, first_flat)
        amplitudes /= numpy.sum(first_flat**2)
        variation   = numpy.median(numpy.abs(amplitudes - numpy.median(amplitudes)))
        physical_limit = self.noise_thr*(-self.thresholds[channel])/template.min()
        amp_min        = min(0.8, max(physical_limit, numpy.median(amplitudes) - self.dispersion[0]*variation))
        amp_max        = max(1.2, numpy.median(amplitudes) + self.dispersion[1]*variation)

        return numpy.array([amp_min, amp_max], dtype=numpy.float32)

    def _center_template(self, template, key):
        if key == 'negative':
            tmpidx = divmod(template.argmin(), template.shape[1])
        elif key == 'positive':
            tmpidx = divmod(template.argmax(), template.shape[1])

        shift            = self._width - tmpidx[1]
        aligned_template = numpy.zeros(template.shape, dtype=numpy.float32)
        if shift > 0:
            aligned_template[:, shift:] = template[:, :-shift]
        elif shift < 0:
            aligned_template[:, :shift] = template[:, -shift:]
        else:
            aligned_template = template
        return aligned_template

    def _reset_data_structures(self, key, channel):
        self.pca_data[key][channel] = numpy.zeros((0, self.pcs.shape[1], len(self.probe.edges[channel])), dtype=numpy.float32)
        self.raw_data[key][channel] = numpy.zeros((0, self._spike_width_, len(self.probe.edges[channel])), dtype=numpy.float32)
        self.clusters[key][channel] = numpy.zeros(0, dtype=numpy.int32)
        self.templates['dat'][key][channel] = numpy.zeros((0, len(self.probe.edges[channel]), self._spike_width_), dtype=numpy.float32)
        self.templates['amp'][key][channel] = numpy.zeros((0, 2), dtype=numpy.float32)

    def _process(self):

        batch = self.inputs['data'].receive()
        peaks = self.inputs['peaks'].receive(blocking=False)
        self.thresholds = self.inputs['mads'].receive(blocking=False)
        if self.receive_pcs:
            self.pcs = self.inputs['pcs'].receive(blocking=False)

        if self.pcs is not None:

            if self.receive_pcs:
                self.log.info("{n} receives the PCA matrices".format(n=self.name_and_counter))
                self.receive_pcs = False
                self._init_data_structures()
                self._set_active_mode()

            if (peaks is not None) and (self.thresholds is not None):

                self.to_reset = []

                while peaks.pop('offset')/self.nb_samples < self.counter:
                    peaks = self.inputs['peaks'].receive()

                if peaks is not None:

                    all_peaks = self._get_all_valid_peaks(peaks)

                    for key in self.sign_peaks:
                        while len(all_peaks[key]) > 0:
                            peak            = all_peaks[key][0]
                            all_peaks[key]  = self._remove_nn_peaks(peak, all_peaks[key])
                            channel, is_neg = self._get_best_channel(batch, key, peak, peaks)
                            waveforms       = self._get_snippet(batch, channel, peak, is_neg).T
                            projection      = numpy.dot(self.pcs[0], waveforms)
                            projection      = projection.reshape(1, projection.shape[0], projection.shape[1])
                            waveforms       = waveforms.reshape(1, waveforms.shape[0], waveforms.shape[1])
                            if is_neg:
                                key = 'negative'
                            else:
                                key = 'positive'

                            self.pca_data[key][channel] = numpy.vstack((self.pca_data[key][channel], projection))
                            self.raw_data[key][channel] = numpy.vstack((self.raw_data[key][channel], waveforms))

                        for channel in xrange(self.nb_channels):
                            if len(self.pca_data[key][channel]) >= self.nb_waveforms:
                                self._perform_clustering(key, channel)

                    if len(self.to_reset) > 0:
                        self.outputs['templates'].send(self.templates)
                        for key, channel in self.to_reset:
                            self._reset_data_structures(key, channel)
        return