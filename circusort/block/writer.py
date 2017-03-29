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
              'nb_buffers' : 1000, 
              't_start'    : None,
              't_comp'     : None}

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

        # self.log.debug("process") # commented to reduce logging

        # TODO receive first batch of data
        batch = self.input.receive()
        # TODO set batch of data to the output
        batch = batch.tobytes()
        self.file.write(batch)

        return

    def _run(self):
        '''TODO add dosctring'''

        while self.running:
            self._process()

        return
