from .block import Block
import numpy
import scipy.interpolate
from circusort.config.probe import Probe


class Density_clustering(Block):
    '''TODO add docstring'''

    name = "Density Clustering"

    params = {'alignment'     : True,
              'time_constant' : 1.,
              'sampling_rate' : 20000.,
              'spike_width'   : 5,
              'nb_waveforms'  : 10000, 
              'probe'         : None,
              'radius'        : None}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        if self.probe == None:
            self.log.error('{n}: the probe file must be specified!'.format(n=self.name))
        else:
            self.probe = Probe(self.probe, radius=self.radius, logger=self.log)
            self.log.info('{n} reads the probe layout'.format(n=self.name))
        self.add_input('data')
        self.add_input('pcs')
        self.add_input('peaks', 'dict')

    def _initialize(self):

        self._spike_width_ = int(self.sampling_rate*self.spike_width*1e-3)
        self.sign_peaks    = None
        self.receive_pcs   = True
        if numpy.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width = (self._spike_width_-1)//2

        if self.alignment:
            self.cdata = numpy.linspace(-self._width, self._width, 5*self._spike_width_)
            self.xdata = numpy.arange(-2*self._width, 2*self._width + 1)

        # self.inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
        # self.inv_nodes[nodes] = numpy.argsort(nodes)

        return

    @property
    def nb_channels(self):
        return self.inputs['data'].shape[0]

    @property
    def nb_samples(self):
        return self.inputs['data'].shape[1]

    def _get_best_channel(self, batch, peak, key):
        if key == 'negative':
            channel = numpy.argmin(batch[:, peak])
            is_neg  = True
        elif key == 'positive':
            channel = numpy.argmax(batch[peak])
            is_neg  = False
        elif key == 'both':
            if numpy.abs(numpy.max(batch[:, peak])) > numpy.abs(numpy.min(batch[:, peak])):
                channel = numpy.argmax(batch[:, peak])
                is_neg  = False
            else:
                channel = numpy.argmin(batch[:, peak])
                is_neg = True
        return channel, is_neg

    def _get_snippet(self, batch, channel, peak, is_neg):
        if self.alignment:
            #idx     = elec_positions[elec]
            #indices = numpy.take(inv_nodes, edges[nodes[i]])
            #elec_positions[i] = numpy.where(indices == i)[0]
            #indices = numpy.take(inv_nodes, edges[nodes[elec]])

            indices = self.probe.edges[channel]
            idx     = self.chan_positions[channel]
            zdata = batch[indices, peak - 2*self._width:peak + 2*self._width + 1]
            ydata = numpy.arange(len(indices))
            if len(ydata) == 1:
                f        = scipy.interpolate.UnivariateSpline(self.xdata, zdata, s=0)
                if is_neg:
                    rmin = (numpy.argmin(f(self.cdata)) - len(self.cdata)/2.)/5.
                else:
                    rmin = (numpy.argmax(f(self.cdata)) - len(self.cdata)/2.)/5.
                ddata    = numpy.linspace(rmin - self._width, rmin + self._width, self._spike_width_)
                sub_mat  = f(ddata).astype(numpy.float32).reshape(self._spike_width_, 1)
            else:
                f        = scipy.interpolate.RectBivariateSpline(self.xdata, ydata, zdata, s=0, ky=min(len(ydata)-1, 3))
                if is_neg:
                    rmin = (numpy.argmin(f(self.cdata, idx)[:, 0]) - len(cdata)/2.)/5.
                else:
                    rmin = (numpy.argmax(f(self.cdata, idx)[:, 0]) - len(cdata)/2.)/5.
                ddata    = numpy.linspace(rmin-self._width, rmin+self._width, self._spike_width_)
                sub_mat  = f(ddata, ydata).astype(numpy.float32)
        else:
            sub_mat = batch[indices, peak - self._width:peak + self._width + 1]

        return sub_mat

    def _guess_output_endpoints(self):

        self.templates  = numpy.zeros(())
        self.decay_time = numpy.exp(-self.nb_samples/float(self.time_constant))
        #self.output.configure(dtype=self.input.dtype, shape=self.input.shape)        
        self.chan_positions = numpy.zeros(self.nb_channels, dtype=numpy.int32)
        for channel in xrange(self.nb_channels):
            #indices = numpy.take(inv_nodes, edges[nodes[i]])
            self.chan_positions[i] = numpy.where(self.probe.edges[channel] == channel)[0]

    def _align(self, batch, channel, peak, key):
        if self.alignment:
            ydata    = batch[channel, peak - 2*self._width:peak + 2*self._width + 1]
            f        = scipy.interpolate.UnivariateSpline(self.xdata, ydata, s=0)
            if key == 'negative':
                rmin = (numpy.argmin(f(self.cdata)) - len(self.cdata)/2.)/5.
            else:
                rmin = (numpy.argmax(f(self.cdata)) - len(self.cdata)/2.)/5.
            ddata    = numpy.linspace(rmin - self._width, rmin + self._width, self._spike_width_)

            result = f(ddata).astype(numpy.float32)
        else:
            result = batch[channel, peak - self._width:peak + self._width + 1]
        return result

    def _process(self):
        if self.receive_pcs:
            self.pcs = self.inputs['pcs'].receive()
            self.log.info("{n} receives the PCA matrices".format(n=self.name))
            self.receive_pcs = False

        batch = self.inputs['data'].receive()
        peaks = self.inputs['peaks'].receive()

        for key in peaks.keys():
            for channel, signed_peaks in peaks[key].items():
                for peak in signed_peaks:
                    channel, is_neg = self._get_best_channel(batch, peak, key)
                    waveforms       = self._get_snippet(batch, channel, peak, is_neg)
                    projection      = numpy.dot(self.pcs[0], waveforms)



        return