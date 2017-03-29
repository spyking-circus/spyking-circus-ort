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

    params = {'nb_samples' : 1024,
              'nb_channels': 10,
              'channels'   : [1, 2, 3], 
              'dtype'      : 'float32'}

    def __init__(self, **kwargs):
        Block.__init__(self, **kwargs)
        self.inputs['data']  = Endpoint(self)
        self.outputs['data'] = Endpoint(self)
        
    def _initialize(self):
        if len(self.channels) > 0:
            nb_channels = len(self.channels)
        else:
            nb_channels = self.nb_channels

        self.output.configure(dtype=self.dtype, shape=(nb_channels, self.nb_samples))
        return

    def _connect(self):
        self.output.socket = self.context.socket(zmq.PAIR)
        self.output.socket.connect(self.output.addr)
        return

    def _run(self):
        while self.running:
            batch = self.input.receive()
            if self.nb_channels > 0:
                batch = batch[self.channels]

            self.output.send(batch.flatten())
        return