from .block import Block
import zmq
import numpy
from scipy import signal
import os
from circusort.base.endpoint import Endpoint
from circusort.base import utils
from circusort.io.generate import synthetic_grid


class Density_clustering(Block):
    '''TODO add docstring'''

    name = "Density Clustering"

    params = {'alignment' : True,
              'time_constant' : 1.}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_input('data')
        self.add_input('peaks', 'dict')

    def _initialize(self):
        return

    @property
    def nb_channels(self):
        return self.inputs['data'].shape[0]

    @property
    def nb_samples(self):
        return self.inputs['data'].shape[1]

    def _guess_output_endpoints(self):

        self.templates  = numpy.zeros(())
        self.decay_time = numpy.exp(-self.nb_samples/float(self.time_constant))
        self.output.configure(dtype=self.input.dtype, shape=self.input.shape)        

    def _process(self):
        peaks = self.inputs['peaks'].receive()
        data  = self.inputs['data'].receive()
        return