import time
import logging

from circusort.base.logger import Logger
from circusort.base.utils import get_log
from circusort.base.process import create_process
from circusort.base.utils import find_interface_address_towards


def create_director(host='127.0.0.1', **kwargs):
    # type: (object, object) -> object
    """Create a new director in this process.

    Parameter:
        host: string (optional)
            The IP address of the host of the director.
    Return:
        director: circusort.base.director
            The director.
    See also:
        circusort.base.director.Director
    """

    interface = find_interface_address_towards(host)
    director = Director(interface, **kwargs)

    return director


class Director(object):

    def __init__(self, host='127.0.0.1', name=None, log_level=logging.INFO):

        # Start logging server
        self.name = name or "Director"
        self.log_level = log_level
        self.logger = Logger(interface=host)
        # Get logger instance
        self.log = get_log(self.logger.address, name=__name__, log_level=self.log_level)

        self.host = host

        self.log.info("{d} is created".format(d=str(self)))
        
        self.managers = {}

    def __del__(self):

        self.log.info("{d} is destroyed".format(d=str(self)))

        for manager in self.managers.itervalues():
            manager.__del__()

    @property
    def nb_managers(self):

        return len(self.managers)

    def get_logger(self):

        return self.logger

    def create_block(self, block_type, name=None, log_level=None, **kwargs):
        # TODO add docstring.

        if self.nb_managers == 0:
            self.create_manager(log_level=self.log_level)

        block = self.get_manager[self.list_managers[0]].create_block(block_type, name, log_level, **kwargs)

        return block

    def create_manager(self, name=None, host=None, log_level=None):
        """Create a new manager process and return a proxy to this process.

        A manager is a process that manages workers.
        """

        if name is None:
            manager_id = 1 + self.nb_managers
            name = "Manager {}".format(manager_id)

        self.log.debug("{d} creates new manager {m}".format(d=str(self), m=name))

        process = create_process(host=host, log_address=self.logger.address, name=name)
        module = process.get_module('circusort.base.manager')
        if log_level is None:
            log_level = self.log_level
        manager = module.Manager(name=name, log_address=self.logger.address, log_level=log_level, host=host)

        self.register_manager(manager)

        return manager

    def register_manager(self, manager):
        
        self.managers.update({manager.name: manager})
        self.log.debug("{d} registers {m}".format(d=str(self), m=manager.name))

        return

    def connect(self, output_endpoints, input_endpoints, protocol=None):
        # TODO add docstring.

        if not type(input_endpoints) == list:
            input_endpoints = [input_endpoints]

        if not type(output_endpoints) == list:
            output_endpoints = [output_endpoints]

        is_local = True
        for input_endpoint in input_endpoints:
            for output_endpoint in output_endpoints:
                if input_endpoint.block.parent != output_endpoint.block.parent:
                    is_local = False
                    break

        for input_endpoint in input_endpoints:
            for output_endpoint in output_endpoints:

                self.log.info("{d} connects {s} to {t}".format(d=str(self), s=output_endpoint.block.name,
                                                               t=input_endpoint.block.name))

                if input_endpoint.block.parent == output_endpoint.block.parent:
                    if protocol is None:
                        if is_local:
                            local_protocol = 'ipc'
                        else:
                            local_protocol = 'tcp'
                    else:
                        local_protocol = protocol
                    self.get_manager(input_endpoint.block.parent).connect(output_endpoint, input_endpoint,
                                                                          local_protocol, show_log=False)
                else:
                    if protocol is None:
                        local_protocol = 'tcp'
                    else:
                        local_protocol = protocol
                    assert local_protocol in ['tcp'], self.log.error('Invalid connection')

                    output_endpoint.initialize(protocol=local_protocol, host=output_endpoint.block.host)
                    description = output_endpoint.get_description()
                    input_endpoint.configure(**description)
                    input_endpoint.block.connect(input_endpoint.name)
                    input_endpoint.block.guess_output_endpoints()

                    string = "{p} connection established from {a}[{s}] to {b}[{t}]"
                    message = string.format(p=local_protocol, s=(output_endpoint.name, output_endpoint.structure),
                                            t=(input_endpoint.name, input_endpoint.structure),
                                            a=output_endpoint.block.name, b=input_endpoint.block.name)
                    self.log.debug(message)

        return

    def initialize(self):

        self.log.info("{d} initializes {s}".format(d=str(self), s=", ".join(self.list_managers())))
        for manager in self.managers.itervalues():
            manager.initialize()

        return

    def start(self, nb_steps=None):

        if nb_steps is None:
            string = "{d} starts {s}"
            message = string.format(d=str(self), s=", ".join(self.list_managers()))
            self.log.info(message)
        else:
            string = "{d} runs {s} for {n} steps"
            message = string.format(d=str(self), s=", ".join(self.list_managers()), n=nb_steps)
            self.log.info(message)

        for manager in self.managers.itervalues():
            manager.start(nb_steps)

        return

    def sleep(self, duration=None):

        self.log.info("{d} sleeps for {k} sec".format(d=str(self), k=duration))
        time.sleep(duration)

        return

    def stop(self):

        self.log.debug("{d} stops {s}".format(d=str(self), s=", ".join(self.list_managers())))
        for manager in self.managers.itervalues():
            manager.stop()
        self.log.info("{d} stops {s}".format(d=str(self), s=", ".join(self.list_managers())))

        return

    def join(self):

        self.log.debug("{d} joins {s}".format(d=str(self), s=", ".join(self.list_managers())))
        for manager in self.managers.itervalues():
            manager.join()
        self.log.info("{d} joins {s}".format(d=str(self), s=", ".join(self.list_managers())))

        return

    def destroy(self):

        self.__del__()

        return

    def __str__(self):

        return "{d}[{i}]".format(d=self.name, i=self.host)

    def list_managers(self):

        return self.managers.keys()

    def get_manager(self, key):

        assert key in self.list_managers(), self.log.warning("%s is not a valid manager" %key)
        return self.managers[key]
