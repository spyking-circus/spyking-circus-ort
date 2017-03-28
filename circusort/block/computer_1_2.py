import numpy
import threading
import zmq

from circusort.base.endpoint import Endpoint
from circusort.base import utils



class Computer_1_2(threading.Thread):
    '''TODO add docstring'''

    def __init__(self, log_address=None):

        threading.Thread.__init__(self)

        self.log_address = log_address

        self.name = "Computer 1 & 2"
        if self.log_address is None:
            raise NotImplementedError("no logger address")
        self.log = utils.get_log(self.log_address, name=__name__)

        self.nb_buffers = 1000

        self.context = zmq.Context()
        self.input = Endpoint(self)
        self.output = Endpoint(self)

        self.log.info("computer 1 & 2 created")

    def initialize(self):
        '''TODO add docstring'''

        self.log.debug("initialization")

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

    def connect(self):
        '''TODO add docstring'''

        self.log.debug("connection")

        self.output.socket = self.context.socket(zmq.PAIR)
        self.output.socket.connect(self.output.addr)

        return

    def configure(self):
        '''TODO add docstring'''

        self.log.debug("configuration")

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

    def run(self):
        '''TODO add dosctring'''

        self.log.debug("run")

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
