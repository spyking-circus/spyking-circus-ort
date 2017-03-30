from .block import Block
import zmq
import numpy
import os
from circusort.base.endpoint import Endpoint
from circusort.base import utils


class Noise_generator(Block):
    '''TODO add docstring'''

    name = "Noise Generator"

    params = {'dtype'         : 'float32',
              'nb_channels'   : 10,
              'sampling_rate' : 20000, 
              'nb_samples'    : 1024}

    def __init__(self, **kwargs):
        Block.__init__(self, **kwargs)
        self.add_output('data')

    def _initialize(self):
        self.output.configure(dtype=self.dtype, shape=(self.nb_channels, self.nb_samples))
        return

    def _connect(self, key):
        self.get_output(key).socket = self.context.socket(zmq.PAIR)
        self.get_output(key).socket.connect(self.get_output(key).addr)
        return

    def _process(self):
        batch = numpy.random.randn(self.nb_channels, self.nb_samples).astype(self.dtype)
        self.output.send(batch)
        return
