from .block import Block
import zmq
import numpy
from scipy import signal
import os
from circusort.base import utils
from circusort.io.generate import synthetic_grid


class Pca(Block):
    '''TODO add docstring'''

    name   = "PCA"

    params = {'spike_width'   : 5,
              'output_dim'    : 5, 
              'alignement'    : True, 
              'nb_waveforms'  : 10000, 
              'sign_peaks'    : 'negative', 
              'sampling_rate' : 20000}

    def __init__(self, **kwargs):
        Block.__init__(self, **kwargs)
        self.add_output('pcs')
        self.add_input('data')
        self.add_input('peaks', 'dict')

    @property
    def nb_channels(self):
        return self.inputs['data'].shape[0]

    @property
    def nb_samples(self):
        return self.inputs['data'].shape[1]

    def _initialize(self):
        self._spike_width_ = int(self.sampling_rate*self.spike_width*1e-3)
        self.count = 0
        if numpy.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self.width        = (self._spike_width_-1)//2
        return

    def _guess_output_endpoints(self):
        self.output.configure(dtype=self.inputs['data'].dtype, shape=(self.nb_channels, self.nb_samples))
        self.waveforms = numpy.zeros((self.nb_waveforms, self._spike_width_), dtype=numpy.float32)

    def _process(self):
        peaks = self.inputs['peaks'].receive()
        batch = self.inputs['data'].receive()
        # print peaks, self.count
        # for channel, peak in peaks.items():
        #     self.waveforms[count] = batch[channel, peak-self.width:peak + self.width + 1]
        #     self.count =+ 1
        #     if self.count == self.nb_waveforms:
        #         break

        #self.output.send(batch.flatten())
        return