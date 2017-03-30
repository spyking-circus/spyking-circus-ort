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
        self.inputs['data']        = Endpoint(self)
        self.outputs['data']       = Endpoint(self)
        self.outputs['thresholds'] = Endpoint(self)

    def _initialize(self):
        self.mads  = numpy.zeros(self.nb_channels)
        self.means = numpy.zeros(self.nb_channels)
        return

    @property
    def nb_channels(self):
        if self.input.shape is not None:
            return self.input.shape[0]
        else:
            return 0

    @property
    def nb_samples(self):
        if self.input.shape is not None:
            return self.input.shape[1]
        else:
            return 0

    def _connect(self, key):
        self.get_output(key).socket = self.context.socket(zmq.PAIR)
        self.get_output(key).socket.connect(self.get_output(key).addr)
        return

    def _process(self):
        
        batch = self.input.receive()
            
        self.means = self.means*self.decay_time + numpy.mean(batch, 0)
        self.output.send(batch.flatten())
        return