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

        self.daemon = True
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
        self.params.update(kwargs)

        self.configure(**self.params)

        self.log.info("{n} has been created".format(n=self.name))
        self.log.debug(str(self))


    def set_manager(self, manager_name):
        self.parent = manager_name

    def _configure(self):
        return

    def initialize(self):

        self.log.debug("{n} is initialized".format(n=self.name))
        self.ready = True
        return self._initialize()

    @property
    def input(self):
        if len(self.inputs) == 1:
            return self.inputs[self.inputs.keys()[0]]
        elif len(self.inputs) == 0:
            self.log.error('No Inputs')
        else:
            self.log.error('Multiple Inputs')

    @property
    def output(self):
        if len(self.outputs) == 1:
            return self.outputs[self.outputs.keys()[0]]
        elif len(self.outputs) == 0:
            self.log.error('No Outputs')
        else:
            self.log.error('Multiple Outputs')

    @property
    def nb_inputs(self):
        return len(self.inputs)

    @property
    def nb_outputs(self):
        return len(self.outputs)

    def get_input(self, key):
        return self.inputs[key]

    def get_output(self, key):
        return self.outputs[key]

    def connect(self, **kwargs):
        '''TODO add docstring'''

        self.log.debug("{n} establishes connections".format(n=self.name))
        return self._connect(**kwargs)

    def configure(self, **kwargs):
        '''TODO add docstring'''

        for key, value in kwargs.items():
            self.params[key] = kwargs[key]
            self.__setattr__(key, value)

        self.log.debug("{n} is configured".format(n=self.name))
        self.ready = False
        #self._configure()
        return

    def run(self):
        '''TODO add dosctring'''

        if not self.ready:
            self.initialize()

        self.log.debug("{n} is running".format(n=self.name))
        #self.running = True

        self.running = True
        self._run()

    def stop(self):
        self.running = False
        self.log.debug("{n} is stopped".format(n=self.name))


    def list_parameters(self):
        return self.params.keys()

    def __str__(self):
        res = "Block object %s with params:\n" %self.name
        for key in self.params.keys():
            res += "|%s = %s\n" %(key, str(getattr(self, key)))
        return res
