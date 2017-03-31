import time
import logging

from .logger import Logger
from . import utils

# from circusort.base.process import Process
from circusort.base.process import create_process



class Director(object):

    def __init__(self, interface='127.0.0.1', name=None, log_level=logging.INFO):

        # Start logging server
        self.name = name or "Director"
        self.log_level = log_level
        self.logger = Logger(interface=interface)
        # Get logger instance
        self.log = utils.get_log(self.logger.address, name=__name__, log_level=self.log_level)

        self.interface = interface

        self.log.info("{d} starts".format(d=str(self)))
        
        self.managers = {}

    def __del__(self):
        self.log.info("{d} stops".format(d=str(self)))

    @property
    def nb_managers(self):
        return len(self.managers)

    def get_logger(self):
        return self.logger

    def create_manager(self, name=None, host=None, log_level=None):
        '''Create a new manager process and return a proxy to this process.

        A manager is a process that manages workers.
        '''
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

    def connect(self, output_endpoint, input_endpoint, protocol='tcp'):
        '''TODO add docstring'''

        self.log.info("{d} connects {s} to {t}".format(d=str(self), s=output_endpoint.block.name, t=input_endpoint.block.name))

        if input_endpoint.block.parent == output_endpoint.block.parent:
            self.get_manager(input_endpoint.block.parent).connect(output_endpoint, input_endpoint, protocol)
        else:
            assert protocol in ['tcp'], self.log.error('Invalid connection')

            input_endpoint.initialize(protocol=protocol, host=input_endpoint.block.host)
            
            output_endpoint.configure(addr=input_endpoint.addr)
            input_endpoint.configure(dtype=output_endpoint.dtype,
                                      shape=output_endpoint.shape)

            output_endpoint.block.connect(output_endpoint.name)
            # We need to resolve the case of blocks that are guessing inputs/outputs shape because of connection. This
            # can only be done if connections are made in order, and if we have only one input/output
            input_endpoint.block.guess_output_endpoints()
            self.log.debug("Connection established from {a}[{s}] to {b}[{t}]".format(s=(output_endpoint.name, output_endpoint.dtype, output_endpoint.shape), 
                                                                                            t=(input_endpoint.name, input_endpoint.dtype, input_endpoint.shape), 
                                                                                            a=output_endpoint.block.name,
                                                                                            b=input_endpoint.block.name))
             
        return

    def initialize(self):
        self.log.info("{d} initializes {s}".format(d=str(self), s=", ".join(self.list_managers())))
        for manager in self.managers.itervalues():
            manager.initialize()
        return

    def start(self, nb_steps=None):
        if nb_steps is None:
            self.log.info("{d} starts {s}".format(d=str(self), s=", ".join(self.list_managers())))
        else:
            self.log.info("{d} runs {s} for {n} steps".format(d=str(self), s=", ".join(self.list_managers()), n=nb_steps))
        for manager in self.managers.itervalues():
            manager.start(nb_steps)
        return

    def sleep(self, duration=None):
        self.log.info("{d} sleeps for {k} sec".format(d=str(self), k=duration))
        time.sleep(duration)
        return

    def stop(self):
        self.log.info("{d} stops {s}".format(d=str(self), s=", ".join(self.list_managers())))
        for manager in self.managers.itervalues():
            manager.stop()
        return

    def join(self):
        self.log.info("{d} joins {s}".format(d=str(self), s=", ".join(self.list_managers())))
        for manager in self.managers.itervalues():
            manager.join()
        return

    def destroy_all(self):
        return

    def __str__(self):
        return "{d}[{i}]".format(d=self.name, i=self.interface)

    def list_managers(self):
        return self.managers.keys()

    def get_manager(self, key):
        assert key in self.list_managers(), self.log.warning("%s is not a valid manager" %key)
        return self.managers[key]
