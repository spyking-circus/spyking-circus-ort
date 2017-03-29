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

        self.log.info("start director {d}".format(d=str(self)))
        
        self.managers = {}

    def __del__(self):
        self.log.info("stop director {d}".format(d=str(self)))

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
            name = "Manager_{}".format(manager_id)

        self.log.debug("{d} creates new manager {m}".format(d=str(self), m=name))

        process = create_process(host=host, log_address=self.logger.address, name=name)
        module = process.get_module('circusort.block.manager')
        if log_level is None:
            log_level = self.log_level
        manager = module.Manager(name=name, log_address=self.logger.address, log_level=log_level, host=host)

        self.register_manager(manager)

        return manager

    def register_manager(self, manager):
        
        self.managers.update({manager.name: manager})
        self.log.debug("{d} registers {m}".format(d=str(self), m=manager.name))
        return

    # def connect(self, input_endpoint, output_endpoint, method='tcp'):
    #     '''TODO add docstring'''

    #     self.log.info("{d} connects couple of blocks".format(d=str(self)))

    #     assert method in ['tcp', 'udp'], self.log.warning('Invalid connection')

        

    #     input_endpoint.configure(addr=output_endpoint.addr)
    #     output_endpoint.configure(dtype=input_endpoint.dtype,
    #                               shape=input_endpoint.shape)

    #     return

    def initialize(self):
        for manager in self.managers.itervalues():
            manager.initialize()
        return

    def start(self):
        for manager in self.managers.itervalues():
            manager.start()
        return

    def sleep(self, duration=None):
        self.log.debug("{d} sleeps {k} sec".format(d=str(self), k=duration))
        time.sleep(duration)
        return

    def stop(self):
        for manager in self.managers.itervalues():
            manager.stop()
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
