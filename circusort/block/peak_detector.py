from .block import Block
import zmq
import numpy
from scipy import signal
import os
from circusort.base.endpoint import Endpoint
from circusort.base import utils
from circusort.io.generate import synthetic_grid


class Peak_detector(Block):
    '''TODO add docstring'''

    name = "Peak detector"

    params = {'time_constant' : True}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        #self.add_output('peaks')
        self.add_input('thresholds')
        self.add_input('data')

    def _initialize(self):
        return

    @property
    def nb_channels(self):
        return self.input.shape[0]
        
    @property
    def nb_samples(self):
        return self.input.shape[1]

    # def _guess_output_endpoints(self):
    #     self.mads  = numpy.zeros(self.nb_channels, dtype=numpy.float32)
    #     self.means = numpy.zeros(self.nb_channels, dtype=numpy.float32)
    #     self.decay_time = numpy.exp(-self.input.shape[1]/self.time_constant)
    #     self.outputs['data'].configure(dtype=self.input.dtype, shape=self.input.shape)
    #     self.outputs['thresholds'].configure(dtype=self.input.dtype, shape=(self.nb_channels, 1))

    def _process(self):
        batch      = self.get_input('data').receive()
        thresholds = self.get_input('thresholds').receive()
        print thresholds
        return