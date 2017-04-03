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

    params = {'cut_off'       : 500,
              'sampling_rate' : 20000,
              'remove_median' : True}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        #self.add_input('data')
        self.add_input('peaks', 'dict')
        #self.add_input('thresholds')

    def _initialize(self):
        return

    @property
    def nb_channels(self):
        return self.input.shape[0]

    @property
    def nb_samples(self):
        return self.input.shape[1]

    def _guess_output_endpoints(self):
        self.output.configure(dtype=self.input.dtype, shape=self.input.shape)        

    def _process(self):
        peaks = self.input.receive()
        print peaks
        return