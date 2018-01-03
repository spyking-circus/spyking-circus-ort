import threading
import zmq
import logging
import time

from circusort.base.endpoint import Endpoint, EOCError
from circusort.base.utils import get_log
from circusort.io.time_measurements import save_time_measurements


class Block(threading.Thread):
    """Block base class."""
    # TODO complete docstring.

    name = "Block"
    params = {}
    inputs = {}
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

        self.log = get_log(self.log_address, name=__name__, log_level=self.log_level)

        self.parent = None
        self.host = None

        self.running = False
        self.ready = False
        self.stop_pending = False
        self.t_start = None
        self.nb_steps = None
        self.is_active = False
        self.start_steps = None
        self.check_interval = 100
        self._measured_times = {}
        self.counter = 0
        self.mpl_display = False
        self.introspection_path = None

        self.context = zmq.Context()
        self.params.update(kwargs)

        self.configure(**self.params)

        self.log.debug("{n} has been created".format(n=self.name))
        self.log.debug(str(self))

    def __del__(self):

        string = "{} is destroyed"
        message = string.format(self.name)
        self.log.debug(message)

    def set_manager(self, manager_name):

        self.parent = manager_name

    def set_host(self, host):

        self.host = host

    def add_output(self, name, structure='array'):

        self.outputs[name] = Endpoint(self, name, structure)

    def add_input(self, name, structure='array'):

        self.inputs[name] = Endpoint(self, name, structure)

    def _initialize(self):

        raise NotImplementedError()

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
            error_msg = "{n} has multiple Inputs:{i}, you must be more explicit"
            self.log.error(error_msg.format(n=self.name, i=self.inputs.keys()))

    @property
    def output(self):

        if len(self.outputs) == 1:
            return self.outputs[self.outputs.keys()[0]]
        elif len(self.outputs) == 0:
            self.log.error('{n} has no Outputs'.format(n=self.name))
        else:
            error_msg = "{n} has multiple Outputs:{o}, you must be more explicit"
            self.log.error(error_msg.format(n=self.name, o=self.outputs.keys()))

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

        string = "{} establishes connections"
        message = string.format(self.name)
        self.log.debug(message)

        self.get_input(key).socket = self.context.socket(zmq.SUB)
        self.get_input(key).socket.connect(self.get_input(key).addr)
        self.get_input(key).socket.setsockopt(zmq.SUBSCRIBE, "")

        return

    def configure(self, **kwargs):

        for key, value in kwargs.items():
            self.params[key] = kwargs[key]
            self.__setattr__(key, value)
        self.ready = False

        string = "{} is configured"
        message = string.format(self.name)
        self.log.debug(message)

        return

    def _guess_output_endpoints(self, **kwargs):

        return

    def guess_output_endpoints(self, **kwargs):

        if self.nb_inputs > 0 and self.nb_outputs > 0:
            string = "{} guesses output connections"
            message = string.format(self.name)
            self.log.debug(message)
            self._guess_output_endpoints(**kwargs)

        return

    def _sync_buffer(self, dictionary, nb_samples):

        offset = dictionary['offset']
        is_synced = self.counter * nb_samples <= offset

        return is_synced

    def run(self):

        if not self.ready:
            self.initialize()

        message = "{} is running".format(self.name)
        self.log.debug(message)

        self.running = True
        self._set_start_step()

        if self.nb_steps is not None:
            while self.counter < self.nb_steps:
                self._process()
                self.counter += 1
                # if numpy.mod(self.counter, self.check_interval) == 0:
                #     self._check_real_time_ratio()
        else:
            try:
                while self.running and not self.stop_pending:
                    self._process()
                    self.counter += 1
                    # if numpy.mod(self.counter, self.check_interval) == 0:
                    #     self._check_real_time_ratio()
            except EOCError:
                # TODO understand why it happens (should not happen).
                for output in self.outputs.itervalues():
                    output.send_end_connection()
                self.stop_pending = True
                self.running = False
            if self.running and self.stop_pending and self.nb_inputs == 0:
                # In this condition, the block is a source block.
                for output in self.outputs.itervalues():
                    output.send_end_connection()
                self.running = False
            try:
                while self.running and self.stop_pending:
                    self._process()
                    self.counter += 1
                    # if numpy.mod(self.counter, self.check_interval) == 0:
                    #     self._check_real_time_ratio()
            except EOCError:
                for output in self.outputs.itervalues():
                    output.send_end_connection()
                self.running = False

        message = "{} is stopped".format(self.name)
        self.log.debug(message)

        self._introspect()
        if self.introspection_path is not None:
            self._save_introspection()

        return

    def _process(self):
        # TODO add docstring.

        raise NotImplementedError()

    def _introspect(self):
        # TODO add docstring.

        nb_buffers = self.counter - self.start_step
        if self.real_time_ratio is not None:
            string = "{} processed {} buffers [{} x real time]"
            message = string.format(self.name, nb_buffers, self.real_time_ratio)
            self.log.info(message)
        else:
            string = "{} processed {} buffers"
            message = string.format(self.name, nb_buffers)
            self.log.info(message)

        return

    def _save_introspection(self):
        # TODO add docstring.

        assert self.introspection_path is not None

        name = self.name.lower().replace(' ', '_')
        save_time_measurements(self.introspection_path, self._measured_times, name=name)

        return

    def stop(self):
        """Send a stop signal to the block.

        The block will wait until the termination of the underlying process.

        """

        self.stop_pending = True

        return

    def kill(self):
        """Kill the block.

        The block won't wait until the termination of the underlying process.

        """

        self.running = False

        return

    def _check_real_time_ratio(self):
        # TODO add docstring.

        data = self.real_time_ratio
        if data is not None and data <= 1 and self.is_active:
            string = "{} is lagging, running at {} x real time"
            message = string.format(self.name_and_counter, data)
            self.log.warning(message)

        return

    def list_parameters(self):

        return self.params.keys()

    def _measure_time(self, label='default', frequency=1):

        if self.counter % frequency == 0:
            time_ = time.time()
            try:
                self._measured_times[label].append(time_)
            except KeyError:
                self._measured_times[label] = [time_]

        return

    @property
    def real_time_ratio(self):

        if hasattr(self, 'nb_samples') and hasattr(self, 'sampling_rate'):
            data_time = float(self.counter) * (float(self.nb_samples) / self.sampling_rate)
            ratio = data_time / self.run_time
        else:
            ratio = None

        return ratio

    @property
    def run_time(self):
        """The time elapsed since the start time [s]."""

        time_ = time.time() - self.t_start

        return time_

    @property
    def name_and_counter(self):

        string = "{}[{} steps]".format(self.name, self.counter)

        return string

    def _set_start_step(self):

        self.t_start = time.time()
        self.start_step = self.counter

        return

    def _set_active_mode(self):

        self.is_active = True
        message = "{} is now active".format(self.name_and_counter)
        self.log.debug(message)
        self._set_start_step()

        return

    def __str__(self):

        string = "Block object {} with params {}".format(self.name, self.params)

        return string
