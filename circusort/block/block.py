import numpy
import threading
import zmq
import logging
import time

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

        self.running  = False
        self.ready    = False
        self.t_start  = None
        self.nb_steps = None
        self.is_active      = False
        self.start_steps    = None
        self.check_interval = 100
        self.counter        = 0
        self.mpl_display    = False

        self.context = zmq.Context()
        self.params.update(kwargs)

        self.configure(**self.params)

        self.log.debug("{n} has been created".format(n=self.name))
        self.log.debug(str(self))

    def __del__(self):
        self.log.debug("{n} is destroyed".format(n=self.name))

    def set_manager(self, manager_name):
        self.parent = manager_name

    def set_host(self, host):
        self.host = host

    def add_output(self, name, structure='array'):
        self.outputs[name] = Endpoint(self, name, structure)

    def add_input(self, name, structure='array'):
        self.inputs[name] = Endpoint(self, name, structure)

    def initialize(self):

        self.log.debug("{n} is initialized".format(n=self.name))
        self.ready = True
        self.counter = 0
        return self._initialize()

    @property
    def input(self):
        if len(self.inputs) == 1:
            return self.inputs[self.inputs.keys()[0]]
        elif len(self.inputs) == 0:
            self.log.error('{n} has no Inputs'.format(n=self.name))
        else:
            self.log.error('{n} has multiple Inputs:{i}, you must be more explicit'.format(n=self.name, i=self.inputs.keys()))

    @property
    def output(self):
        if len(self.outputs) == 1:
            return self.outputs[self.outputs.keys()[0]]
        elif len(self.outputs) == 0:
            self.log.error('{n} has no Outputs'.format(n=self.name))
        else:
            self.log.error('{n} has multiple Outputs:{o}, you must be more explicit'.format(n=self.name, o=self.outputs.keys()))

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

    def connect(self, key):
        self.log.debug("{n} establishes connections".format(n=self.name))
        self.get_input(key).socket = self.context.socket(zmq.SUB)
        self.get_input(key).socket.connect(self.get_input(key).addr)
        self.get_input(key).socket.setsockopt(zmq.SUBSCRIBE, "")


    def configure(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = kwargs[key]
            self.__setattr__(key, value)

        self.log.debug("{n} is configured".format(n=self.name))
        self.ready = False
        return

    def guess_output_endpoints(self, **kwargs):
        if self.nb_inputs > 0 and self.nb_outputs > 0:
            self.log.debug("{n} guesses output connections".format(n=self.name))
            return self._guess_output_endpoints(**kwargs)


    def _sync_buffer(self, dict, nb_samples):
        offset = dict['offset']
        if offset < self.counter * nb_samples:
            return False
        return True

    def run(self):
        if not self.ready:
            self.initialize()

        self.log.debug("{n} is running".format(n=self.name))

        self.running = True
        self._set_start_step()

        if self.nb_steps is not None:
            while self.counter < self.nb_steps:
                self._process()
                self.counter += 1
                if numpy.mod(self.counter, self.check_interval) == 0:
                    self._check_real_time_ratio()
        else:
            while self.running:
                self._process()
                self.counter += 1
                if numpy.mod(self.counter, self.check_interval) == 0:
                    self._check_real_time_ratio()


    def stop(self):
        self.running = False
        self.log.debug("{n} is stopped".format(n=self.name))
        if self.real_time_ratio is not None:
            self.log.info("{n} processed {m} buffers [{k} x real time]".format(n=self.name, m=self.counter - self.start_step, k=self.real_time_ratio))
        else:
            self.log.info("{n} processed {m} buffers".format(n=self.name, m=self.counter - self.start_step))


    def _check_real_time_ratio(self):
        data = self.real_time_ratio
        if data is not None and data <= 1 and self.is_active:
            self.log.warning("{n} is lagging, running at {k} x real time".format(n=self.name_and_counter, k=data))

    def list_parameters(self):
        return self.params.keys()

    @property
    def real_time_ratio(self):
        if hasattr(self, 'nb_samples') and hasattr(self, 'sampling_rate'):
            return (self.nb_samples*self.counter/self.sampling_rate)/self.run_time
        else:
            return None

    @property
    def run_time(self):
        return time.time() - self.t_start

    @property
    def name_and_counter(self):
        return "{n}[{k} steps]".format(n=self.name, k=self.counter)

    def _set_start_step(self):
        self.t_start    = time.time()
        self.start_step = self.counter

    def _set_active_mode(self):
        self.is_active = True
        self.log.debug("{n} is now active".format(n=self.name_and_counter))
        self._set_start_step()

    def __str__(self):
        res = "Block object {n} with params {s}".format(n=self.name, s=self.params)
        return res
