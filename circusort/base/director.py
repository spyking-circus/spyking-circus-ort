import collections
import logging
import time

from circusort.base.logger import Logger
from circusort.base.utils import get_log
from circusort.base.process import create_process
from circusort.base.utils import find_interface_address_towards


def create_director(host='127.0.0.1', **kwargs):
    """Create a new director in this process.

    Arguments:
        host: string (optional)
            The IP address of the host of the director.
        kwargs: dictionary (optional)
            Additional keyword arguments.
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
    """Director.

    A director is a block which handles manager blocks.

    Attributes:
        name: string
        log_level: integer
        logger: ...
        log: ...
        host: string
        managers: collections.OrderedDict
    """

    def __init__(self, host='127.0.0.1', name=None, log_level=logging.INFO, log_path=None):
        """Initialize a director.

        Arguments:
            host: string (optional)
                The default value is '127.0.0.1'.
            name: none | string (optional)
                The default value is None.
            log_level: integer (optional)
                The default value is logging.INFO.
            log_path: none | string (optional)
                The default value is None.
        """

        # Start logging server
        self.name = name or "Director"
        self.log_level = log_level
        self.logger = Logger(interface=host, path=log_path)
        # Get logger instance
        self.log = get_log(self.logger.address, name=__name__, log_level=self.log_level)

        self.host = host

        # Log info message.
        string = "{} is created"
        message = string.format(str(self))
        self.log.info(message)
        
        self.managers = collections.OrderedDict()

    def __del__(self):

        # Log info message.
        string = "{} is destroyed"
        message = string.format(str(self))
        self.log.info(message)

        # Delete each manager.
        for manager in self.managers.values():
            manager.__del__()

    @property
    def nb_managers(self):

        return len(self.managers)

    def get_logger(self):

        return self.logger

    def create_block(self, block_type, name=None, log_level=None, **kwargs):

        if self.nb_managers == 0:
            self.create_manager(log_level=self.log_level)

        block = self.get_manager(self.list_managers()[0]).create_block(block_type, name, log_level, **kwargs)

        return block

    def create_manager(self, name=None, host=None, log_level=None):
        """Create a new manager process and return a proxy to this process.

        A manager is a process that manages workers.
        """

        if name is None:
            manager_id = 1 + self.nb_managers
            name = "Manager {}".format(manager_id)

        # Log debug message.
        string = "{} creates new manager {}"
        message = string.format(str(self), name)
        self.log.debug(message)

        if log_level is None:
            log_level = self.log_level
        process = create_process(host=host, log_address=self.logger.address, name=name, log_level=log_level)
        module = process.get_module('circusort.base.manager')
        manager = module.Manager(name=name, log_address=self.logger.address, log_level=log_level, host=host)

        self.register_manager(manager)

        return manager

    def register_manager(self, manager):

        # Update manager.
        self.managers.update({manager.name: manager})

        # Log debug message.
        string = "{} registers {}"
        message = string.format(str(self), manager.name)
        self.log.debug(message)

        return

    def connect(self, output_endpoints, input_endpoints, protocol=None):

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

                # Log info message.
                string = "{} connects {} to {}"
                message = string.format(str(self), output_endpoint.block.name, input_endpoint.block.name)
                self.log.info(message)

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

                    # Create and bind socket.
                    output_endpoint.initialize(protocol=local_protocol, host=output_endpoint.block.host)
                    # Transmit information for socket connection.
                    description = output_endpoint.get_description()
                    input_endpoint.configure(**description)
                    # Connect socket.
                    input_endpoint.block.connect(input_endpoint.name)
                    # Transmit information between blocks.
                    params = output_endpoint.block.get_output_parameters()
                    input_endpoint.block.configure_input_parameters(**params)
                    # Transmit information between endpoints.
                    params = output_endpoint.get_output_parameters()
                    input_endpoint.configure_input_parameters(**params)
                    # Update initialization in output block (if necessary).
                    if input_endpoint.block.input_endpoints_are_configured:
                        input_endpoint.block.update_initialization()

                    # Log debug message.
                    string = "{} connection established from {}[{}] to {}[{}]"
                    message = string.format(
                        local_protocol,
                        output_endpoint.block.name,
                        (output_endpoint.name, output_endpoint.structure),
                        input_endpoint.block.name,
                        (input_endpoint.name, input_endpoint.structure)
                    )
                    self.log.debug(message)

        return

    def connect_network(self, network):

        # Log info message.
        string = "{} connects {} network"
        message = string.format(str(self), network.name)
        self.log.info(message)

        # Connect network.
        network.connect()

        return

    def initialize(self):

        # Log info message.
        string = "{} initializes {}"
        message = string.format(str(self), ", ".join(self.list_managers()))
        self.log.info(message)

        # Initialize each manager.
        for manager in self.managers.values():
            manager.initialize()

        return

    def start(self, nb_steps=None):

        # Log info message.
        if nb_steps is None:
            string = "{d} starts {s}"
            message = string.format(d=str(self), s=", ".join(self.list_managers()))
            self.log.info(message)
        else:
            string = "{d} runs {s} for {n} steps"
            message = string.format(d=str(self), s=", ".join(self.list_managers()), n=nb_steps)
            self.log.info(message)

        # Start each manager.
        for manager in self.managers.values():
            manager.start(nb_steps)

        return

    def sleep(self, duration=None):

        # Log info message.
        string = "{} sleeps for {} sec"
        message = string.format(str(self), duration)
        self.log.info(message)

        # Sleep.
        time.sleep(duration)

        return

    def stop(self):

        # Log debug message.
        string = "{} stops {}"
        message = string.format(str(self), ", ".join(self.list_managers()))
        self.log.debug(message)

        # Stop each manager.
        for manager in self.managers.values():
            manager.stop()

        # Log info message.
        string = "{} stopped {}"
        message = string.format(str(self), ", ".join(self.list_managers()))
        self.log.info(message)

        return

    def join(self):

        # Log debug message.
        string = "{} joins {}"
        message = string.format(str(self), ", ".join(self.list_managers()))
        self.log.debug(message)

        # Join each manager.
        for manager in self.managers.values():
            manager.join()

        # Log info message.
        string = "{} joins {}"
        message = string.format(str(self), ", ".join(self.list_managers()))
        self.log.info(message)

        return

    def destroy(self):

        self.__del__()

        return

    def __str__(self):

        string = "{}[{}]".format(self.name, self.host)

        return string

    def list_managers(self):

        return list(self.managers.keys())

    def get_manager(self, key):

        assert key in self.list_managers(), self.log.warning("{} is not a valid manager".format(key))

        return self.managers[key]
