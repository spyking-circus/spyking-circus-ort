from .block import Block
import zmq
import numpy
import os


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

    def _process(self):
        batch = numpy.random.randn(self.nb_channels, self.nb_samples).astype(self.dtype)
        self.output.send(batch)
        return
