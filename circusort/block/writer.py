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


    inputs = {'data' : None}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        self.inputs['data'] = Endpoint(self)
        self.t_start        = None
        self.t_comp         = None
        #self.input          = Endpoint(self)

    def _initialize(self):
        '''TODO add docstring'''

        # TODO create output file object
        transport = 'tcp'
        host = '127.0.0.1'
        port = '*'
        endpoint = '{h}:{p}'.format(h=host, p=port)
        address = '{t}://{e}'.format(t=transport, e=endpoint)
        self.get_input('data').socket = self.context.socket(zmq.PAIR)
        # self.input.socket.setsockopt(zmq.RCVTIMEO, 10000)
        self.get_input('data').socket.bind(address)
        self.get_input('data').addr = self.get_input('data').socket.getsockopt(zmq.LAST_ENDPOINT)
        self.file = open(self.data_path, mode='wb')

        return

    def _connect(self):
        '''TODO add docstring'''
        return

    def _process(self):

        # self.log.debug("process") # commented to reduce logging

        # TODO receive first batch of data
        batch = self.get_input('data').receive()

        # TODO set batch of data to the output
        batch = batch.tobytes()
        self.file.write(batch)

        return

    def _run(self):
        '''TODO add dosctring'''

        self.log.debug("run")

        self._process()

        self.t_start = time.time()

        for i in range(1, self.nb_buffers):
            self._process()

        self.t_comp = time.time() - self.t_start

        return
