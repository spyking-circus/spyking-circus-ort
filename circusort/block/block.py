import numpy
import threading
import zmq
import logging

from circusort.base.endpoint import Endpoint
from circusort.base import utils


class Block(threading.Thread):
    '''TODO add docstring'''

    name    = "Block"
    params  = {}
    inputs  = {}
    outputs = {}

    def __init__(self, name=None, log_address=None, log_level=logging.INFO, **kwargs):

        threading.Thread.__init__(self)

        self.log_address = log_address
        self.log_level = log_level
        if name is not None:
            self.name = name

        if self.log_address is None:
            raise NotImplementedError("no logger address")

        self.log = utils.get_log(self.log_address, name=__name__, log_level=self.log_level)

        self.running = False
        self.ready   = False
        self.context = zmq.Context()

        self.configure(**kwargs)

        self.log.info("{n} has been created".format(n=self.name))

    def __getattr__(self, key):
        return self.params[key]

    def initialize(self):

        self.log.debug("{n} is initialized".format(n=self.name))
        self.ready = True
        return self._initialize()

    def connect(self, **kwargs):
        '''TODO add docstring'''

        self.log.debug("{n} establishes connections".format(n=self.name))
        return self._connect(**kwargs)

    def configure(self, **kwargs):
        '''TODO add docstring'''

        for key in self.params.keys():
            if key in kwargs.keys():
                self.params[key] = kwargs[key]

        self.log.debug("{n} is configured".format(n=self.name))
        self.ready = False
        return

    def run(self):
        '''TODO add dosctring'''

        if not self.ready:
            self.initialize()

        self.log.debug("run")
        #self.running = True

        #while self.running:
        self._run()

    def stop(self):
        self.running = False

    def list_parameters(self):
        return self.params.keys()

    def __str__(self):
        res = "Block object %s with params:\n"
        for key, value in self.params.items():
            res += "|%s = %s\n" %(key, str(value))
        return res
