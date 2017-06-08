from .block import Block
import numpy
import scipy.interpolate
from circusort.utils.algorithms import PCAEstimator

class Pca(Block):
    '''TODO add docstring'''

    name   = "PCA"

    params = {'spike_width'   : 5,
              'output_dim'    : 5, 
              'alignment'     : True,
              'nb_waveforms'  : 10000,
              'sampling_rate' : 20000}

    def __init__(self, **kwargs):
        Block.__init__(self, **kwargs)
        self.add_output('pcs')
        self.add_input('data')
        self.add_input('peaks')

    @property
    def nb_channels(self):
        return self.inputs['data'].shape[1]

    @property
    def nb_samples(self):
        return self.inputs['data'].shape[0]

    def _initialize(self):
        self._spike_width_ = int(self.sampling_rate*self.spike_width*1e-3)
        self.sign_peaks    = None
        self.send_pcs      = True
        if numpy.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width = (self._spike_width_-1)//2

        if self.alignment:
            self.cdata = numpy.linspace(-self._width, self._width, 5*self._spike_width_)
            self.xdata = numpy.arange(-2*self._width, 2*self._width + 1)

        return

    def _guess_output_endpoints(self):
        self.outputs['pcs'].configure(dtype='float32', shape=(2, self.output_dim, self._spike_width_))
        self.pcs = numpy.zeros((2, self.output_dim, self._spike_width_), dtype=numpy.float32)

    def _is_valid(self, peak):
        if self.alignment:
            return (peak >= 2*self._width) and (peak + 2*self._width < self.nb_samples)
        else:
            return (peak >= self._width) and (peak + self._width < self.nb_samples)

    def is_ready(self, key=None):
        if key is not None:
            return (self.nb_spikes[key] == self.nb_waveforms) and not self.has_pcs[key]
        else:
            return bool(numpy.prod([i for i in self.has_pcs.values()]))

    def _get_waveform(self, batch, channel, peak, key):
        if self.alignment:
            ydata    = batch[peak - 2*self._width:peak + 2*self._width + 1, channel]
            f        = scipy.interpolate.UnivariateSpline(self.xdata, ydata, s=0)
            if key == 'negative':
                rmin = (numpy.argmin(f(self.cdata)) - len(self.cdata)/2.)/5.
            else:
                rmin = (numpy.argmax(f(self.cdata)) - len(self.cdata)/2.)/5.
            ddata    = numpy.linspace(rmin - self._width, rmin + self._width, self._spike_width_)

            result = f(ddata).astype(numpy.float32)
        else:
            result = batch[peak - self._width:peak + self._width + 1, channel]
        return result

    def _infer_sign_peaks(self, peaks):
        self.sign_peaks = [str(i) for i in peaks.keys()]

        self.nb_spikes = {}
        self.waveforms = {}
        self.has_pcs   = {}
        for key in peaks.keys():
            self.nb_spikes[key] = 0
            self.has_pcs[key]   = False
            self.waveforms[key] = numpy.zeros((self.nb_waveforms, self._spike_width_), dtype=numpy.float32)

    def _process(self):

        batch = self.inputs['data'].receive()
        peaks = self.inputs['peaks'].receive(blocking=False)
        
        if peaks is not None:

            while peaks.pop('offset')/self.nb_samples < self.counter:
                peaks = self.inputs['peaks'].receive()
            
            if self.sign_peaks is None:
                self._infer_sign_peaks(peaks)
            self._set_active_mode()

            if self.send_pcs:

                for key in self.sign_peaks:
                    for channel, signed_peaks in peaks[key].items():
                        if self.nb_spikes[key] < self.nb_waveforms:
                            for peak in signed_peaks:
                                if self.nb_spikes[key] < self.nb_waveforms and self._is_valid(peak):
                                    self.waveforms[key][self.nb_spikes[key]] = self._get_waveform(batch, int(channel), peak, key)
                                    self.nb_spikes[key] += 1

                    if self.is_ready(key):
                        self.log.info("{n} computes the PCA matrix for {m} spikes".format(n=self.name_and_counter, m=key))
                        pca          = PCAEstimator(self.output_dim, copy=False)
                        #numpy.save('pca_%s' %key, self.waveforms[key])
                        res_pca      = pca.fit_transform(self.waveforms[key].T).astype(numpy.float32)
                        if key == 'negative':
                            self.pcs[0] = res_pca.T
                        elif key == 'positive':
                            self.pcs[1] = res_pca.T
                        self.has_pcs[key] = True

            if self.is_ready() and self.send_pcs:
                self.outputs['pcs'].send(self.pcs)
                self.send_pcs = False
        return