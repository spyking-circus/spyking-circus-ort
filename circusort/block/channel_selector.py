from .block import Block
import zmq
import numpy
from scipy import signal
import os
from circusort.base.endpoint import Endpoint
from circusort.base import utils
from circusort.io.generate import synthetic_grid


class Channel_selector(Block):
    '''TODO add docstring'''

    name   = "Channel Selector"

    params = {'channels'   : [], 
              'dtype'      : 'float32'}

    def __init__(self, **kwargs):
        Block.__init__(self, **kwargs)
        self.inputs['data']  = Endpoint(self)
        self.outputs['data'] = Endpoint(self)
        
    @property
    def nb_channels(self):
        if self.input.shape is not None:
            if len(self.channels) > 0:
                nb_channels = len(self.channels)
            else:
                nb_channels = self.input.shape[0]
            return nb_channels
        else:
            return 0

    @property
    def nb_samples(self):
        if self.input.shape is not None:
            return self.input.shape[1]
        else:
            return 0

    def _initialize(self):
        return

    def _guess_output_endpoints(self):
        self.output.configure(dtype=self.dtype, shape=(self.nb_channels, self.nb_samples))

    def _connect(self):
        self.output.socket = self.context.socket(zmq.PAIR)
        self.output.socket.connect(self.output.addr)
        return

    def _process(self):
        batch = self.input.receive()
        if len(self.channels) > 0:
            batch = batch[self.channels]

        self.output.send(batch.flatten())
        return