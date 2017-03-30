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

    params = {'time_constant' : True}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('data')
        self.add_output('thresholds')
        self.add_input('data')

    def _initialize(self):
        return

    @property
    def nb_channels(self):
        return self.input.shape[0]
        

    @property
    def nb_samples(self):
        return self.input.shape[1]

    def _connect(self, key):
        self.get_output(key).socket = self.context.socket(zmq.PAIR)
        self.get_output(key).socket.connect(self.get_output(key).addr)
        return

    def _guess_output_endpoints(self):
        self.mads  = numpy.zeros(self.nb_channels)
        self.means = numpy.zeros(self.nb_channels)
        self.decay_time = numpy.exp(-self.input.shape[1]/self.time_constant)
        self.outputs['data'].configure(dtype=self.input.dtype, shape=self.input.shape)
        self.outputs['thresholds'].configure(dtype=self.input.dtype, shape=(self.nb_channels, 1))

    def _process(self):
        
        batch      = self.input.receive()
        self.means = self.means*self.decay_time + numpy.mean(batch, 1)
        self.outputs['data'].send(batch.flatten())
        self.outputs['thresholds'].send(self.means.flatten())

        return