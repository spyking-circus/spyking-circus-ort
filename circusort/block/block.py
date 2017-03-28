import numpy
import threading
import zmq
import logging

from circusort.base.endpoint import Endpoint
from circusort.base import utils


class Block(threading.Thread):
    '''TODO add docstring'''

    def __init__(self, name, log_address=None, log_level=logging.INFO):

        threading.Thread.__init__(self)

        self.log_address = log_address
        self.log_level = log_level

        self.name = "Computer 1"
        if self.log_address is None:
            raise NotImplementedError("no logger address")
        self.log = utils.get_log(self.log_address, name=__name__, log_level=self.log_level)

        self.inputs  =  {}
        self.outputs =  {}
        self.running  = False

        self.log.info("computer 1 created")

    def initialize(self, **kwargs):
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

    def configure(self, **kwargs):
        '''TODO add docstring'''

        self.log.debug("configuration")

        return self._configure(**kwags)

    def run(self):
        '''TODO add dosctring'''

        self.log.debug("run")
        self.running = True

        while self.running:
            self._run()


