from .block import Block
import zmq
import numpy
from scipy import signal
import os
from circusort.base.endpoint import Endpoint
from circusort.base import utils
from circusort.io.generate import synthetic_grid


class Mad_estimator(Block):
    '''TODO add docstring'''

    name = "MAD Estimator"

    params = {'time_constant' : 1.}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('mads')
        self.add_input('data')

    def _initialize(self):
        return

    @property
    def nb_channels(self):
        return self.input.shape[0]
        
    @property
    def nb_samples(self):
        return self.input.shape[1]

    def _guess_output_endpoints(self):
        self.mads         = numpy.zeros((self.nb_channels, 1), dtype=numpy.float32)
        self.median_means = numpy.zeros((self.nb_channels, 1), dtype=numpy.float32)
        self.decay_time   = numpy.exp(-self.nb_samples/float(self.time_constant))
        self.outputs['mads'].configure(dtype=self.input.dtype, shape=(self.nb_channels, 1))

    def _process(self):
        batch     = self.input.receive()
        self.median_means[:,0] = self.median_means[:,0]*self.decay_time + numpy.median(batch, 1)
        self.mads = self.mads*self.decay_time + numpy.median(numpy.abs(batch) - self.median_means, 1)
        self.get_output('mads').send(self.median_means.flatten())
        return