from .block import Block
import numpy
import time
import scipy.interpolate
from circusort.config.probe import Probe
from circusort.utils.algorithms import PCAEstimator
from circusort.utils.clustering import rho_estimation, density_based_clustering
from circusort.utils.clustering import OnlineManager

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
              'sub_dim'       : 5,
              'extraction'    : 'median-raw', 
              'two_components': False, 
              'decay_factor'  : 0.35,
              'mu'            : 4,
              'epsilon'       : 0.5,
              'frequency'     : 5000,
              'theta'         : -numpy.log(0.001)}

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
            for channel in xrange(self.nb_channels):
                self.chan_positions[channel] = numpy.where(self.probe.edges[channel] == channel)[0]

    def _init_data_structures(self):
        self.raw_data   = {}
        self.templates  = {}
        self.managers   = {}

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
            self.raw_data[key] = {}
            self.managers[key] = {}
            self.templates['dat'][key] = {}
            self.templates['amp'][key] = {}
            if self.two_components:
                self.templates['two'][key] = {}

            for channel in xrange(self.nb_channels):
                self.managers[key][channel] = OnlineManager(name='OnlineManger for {p} peak on channel {c}'.format(p=key, c=channel), logger=self.log)
                self._reset_data_structures(key, channel)

    def _update_templates(self, templates, amplitudes, key, channel):
       
        for t in templates:
    
            template         = t.reshape(self._spike_width_, len(self.probe.edges[channel])).T
            template, shift  = self._center_template(template, key)
            #amplitudes, amps = self._get_amplitudes(data, template, channel)

            # if self.two_components:

            #     x, y, z     = data.shape
            #     data_flat   = data.reshape(x, y*z)
            #     first_flat  = template.reshape(y*z, 1)

            #     for i in xrange(x):
            #         data_flat[i, :] -= amps[i]*first_flat[:, 0]

            #     if len(data_flat) > 1:
            #         pca       = PCAEstimator(1)
            #         res_pca   = pca.fit_transform(data_flat).astype(numpy.float32)
            #         template2 = pca.components_.T.astype(numpy.float32).reshape(y, z)
            #     else:
            #         template2 = data_flat.reshape(y, z)/numpy.sum(data_flat**2)

            #     template2    = template2.T
            #     template2, _ = self._center_template(template2, key, shift)

            self.templates['dat'][key][channel] = numpy.vstack((self.templates['dat'][key][channel], template.reshape(1, template.shape[0], template.shape[1])))
        
        self.templates['amp'][key][channel] = amplitudes
        if self.two_components:
            self.templates['two'][key][channel] = numpy.vstack((self.templates['two'][key][channel], template2.reshape(1, template2.shape[0], template2.shape[1])))

        self.to_reset += [(key, channel)]

    def _get_amplitudes(self, data, template, channel):
        x, y, z     = data.shape
        data        = data.reshape(x, y*z)
        first_flat  = template.reshape(y*z, 1)
        amplitudes  = numpy.dot(data, first_flat)
        amplitudes /= numpy.sum(first_flat**2)
        variation   = numpy.median(numpy.abs(amplitudes - numpy.median(amplitudes)))
        physical_limit = self.noise_thr*(-self.thresholds[channel])/template[self.chan_positions[channel], self._width]
        amp_min        = min(0.8, max(physical_limit, numpy.median(amplitudes) - self.dispersion[0]*variation))
        amp_max        = max(1.2, numpy.median(amplitudes) + self.dispersion[1]*variation)

        return numpy.array([amp_min, amp_max], dtype=numpy.float32), amplitudes

    def _center_template(self, template, key, shift=None):
        if shift is None:
            if key == 'negative':
                tmpidx = divmod(template.argmin(), template.shape[1])
            elif key == 'positive':
                tmpidx = divmod(template.argmax(), template.shape[1])

            shift = self._width - tmpidx[1]
    
        aligned_template = numpy.zeros(template.shape, dtype=numpy.float32)
        if shift > 0:
            aligned_template[:, shift:] = template[:, :-shift]
        elif shift < 0:
            aligned_template[:, :shift] = template[:, -shift:]
        else:
            aligned_template = template
        return aligned_template, shift

    def _reset_data_structures(self, key, channel):
        self.raw_data[key][channel] = numpy.zeros((0, self._spike_width_ * len(self.probe.edges[channel])), dtype=numpy.float32)
        self.templates['dat'][key][channel] = numpy.zeros((0, len(self.probe.edges[channel]), self._spike_width_), dtype=numpy.float32)
        self.templates['amp'][key][channel] = numpy.zeros((0, 2), dtype=numpy.float32)
        if self.two_components:
            self.templates['two'][key][channel] = numpy.zeros((0, len(self.probe.edges[channel]), self._spike_width_), dtype=numpy.float32)

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
                            waveforms       = waveforms.reshape(1, waveforms.shape[0]*waveforms.shape[1])
                            if is_neg:
                                key = 'negative'
                            else:
                                key = 'positive'

                            if not self.managers[key][channel].is_ready:
                                self.raw_data[key][channel] = numpy.vstack((self.raw_data[key][channel], waveforms))
                            else:
                                self.managers[key][channel].update(self.counter, waveforms)
                           
                        for channel in xrange(self.nb_channels):
                            
                            self.managers[key][channel].set_physical_threshold(self.thresholds[channel])

                            if len(self.raw_data[key][channel]) >= self.nb_waveforms and not self.managers[key][channel].is_ready:
                                templates, amplitudes = self.managers[key][channel].initialize(self.counter, self.raw_data[key][channel])
                                self._update_templates(templates, amplitudes, key, channel)

                            elif self.managers[key][channel].time_to_cluster(self.frequency):
                                templates, amplitudes = self.managers[key][channel].cluster()
                                self._update_templates(templates, amplitudes, key, channel)

                    if len(self.to_reset) > 0:
                        self.outputs['templates'].send(self.templates)
                        for key, channel in self.to_reset:
                            self._reset_data_structures(key, channel)
        return