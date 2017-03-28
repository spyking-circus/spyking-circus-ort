from .block import Block
import zmq
import numpy
from circusort.base.endpoint import Endpoint
from circusort.base import utils



class Computer_1_2(Block):
    '''TODO add docstring'''

    name = "Two operations"

    params = {'data_path'  : '/tmp/output.dat',
              'nb_buffers' : 1000}


    #inputs  = {'data_in' : None}
    #outputs = {'data_out' : None}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.input = Endpoint(self)
        self.output = Endpoint(self)


    def _initialize(self):
        '''TODO add docstring'''

        # Bind socket for input data
        transport = 'tcp'
        host = '127.0.0.1'
        port = '*'
        endpoint = '{h}:{p}'.format(h=host, p=port)
        address = '{t}://{e}'.format(t=transport, e=endpoint)
        self.input.socket = self.context.socket(zmq.PAIR)
        # self.input.socket.setsockopt(zmq.RCVTIMEO, 10000)
        self.input.socket.bind(address)
        self.input.addr = self.input.socket.getsockopt(zmq.LAST_ENDPOINT)

        return

    def _connect(self):
        '''TODO add docstring'''

        self.output.socket = self.context.socket(zmq.PAIR)
        self.output.socket.connect(self.output.addr)

        return

    def _configure(self):
        '''TODO add docstring'''

        self.output.dtype = self.input.dtype
        self.output.shape = self.input.shape

        return

    def operation_1(self, batch, k_1=10):
        '''TODO add docstring'''

        for k in range(0, k_1):
            batch = numpy.add(batch, +1.0)
            batch = numpy.add(batch, -1.0)

        return batch

    def operation_2(self, batch, k_2=10):
        '''TODO add docstring'''

        for k in range(0, k_2):
            batch = numpy.add(batch, -1.0)
            batch = numpy.add(batch, +1.0)

        return batch

    def _run(self):
        '''TODO add dosctring'''

        for i in range(0, self.nb_buffers):
            # a. Receive batch of data
            batch = self.input.receive()
            # b. Apply operation #1
            self.operation_1(batch)
            # c. Apply operation #2
            self.operation_2(batch)
            # d. Send batch of data
            self.output.send(batch)

        return
