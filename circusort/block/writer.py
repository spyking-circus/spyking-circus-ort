from .block import Block
import zmq
import numpy
import time
from circusort.base.endpoint import Endpoint
from circusort.base import utils



class Writer(Block):
    '''TODO add docstring'''

    name   = "File writer"

    params = {'data_path'  : '/tmp/output.dat',
              'nb_buffers' : 1000}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.inputs['data'] = Endpoint(self)


    def _initialize(self):
        '''TODO add docstring'''
        self.file = open(self.data_path, mode='wb')

        return

    def _connect(self):
        '''TODO add docstring'''
        return

    def _process(self):
        batch = self.input.receive()
        batch = batch.tobytes()
        self.file.write(batch)

        return
