import time

from .logger import Logger
from .manager import Manager
from . import utils

from circusort.base.process import Process



class Director(object):

    def __init__(self):

        # Start logging server
        self.logger = Logger()
        # Get logger instance
        self.log = utils.get_log(self.logger.address, name=__name__)

        self.interface = utils.find_ethernet_interface()

        self.log.debug("start director at {i}".format(i=self.interface))

        # TODO remove following line...
        self.name = "Director's name"
        self.managers = {}

    def __del__(self):
        self.log.debug("stop director at {i}".format(i=self.interface))

    @property
    def nb_managers(self):
        return len(self.managers)

    def get_logger(self):
        return self.logger

    def create_manager(self, interface=None):
        '''Create a new manager process and return a proxy to this process.

        A manager is a process that manages workers.
        '''
        self.log.info("director at {i} creates new manager".format(i=self.interface))
        # TODO check if address is local or not...
        # 1. local -> create new process locally (if necessary)            [cli]
        # 2. network -> create new process remotely (if necessary)   [ssh + cli]

        # TODO remove code attempt...
        # process = Process(address=address)
        # module = process.client._import('circusort.base.manager')
        # manager = module.Manager(director=self)

        # TODO remove following test lines...
        #   a. How to deal with local v.s. remote MACHINE?
        #   b. How to deal with local v.s. remote PROCESS?
        mode = 'local'
        if mode == 'local': # run manager inside the local process
            self.log.info("spawn local manager process")
            process = Process(log_address=self.logger.address)
            # or # process = Process(host='localhost')
            # or # process = Process(host='127.0.0.1')
            module = process.get_module('circusort.cli.manager')
            manager = module.Manager()
        elif mode == 'remote': # run manager inside a remote process
            self.log.info("spawn remote manager process")
            process = Process(host='134.157.180.212', log_address=self.logger.address)
            module = process.get_module('circusort.cli.manager')
            manager = module.Manager()
        else:
            raise ValueError("invalid mode value: {m}".format(m=mode))

        # # TODO uncomment following line
        # manager = Manager(interface=interface, log_addr=self.logger.address)
        self.register_manager(manager)

        return manager

    def register_manager(self, manager, name=None):
        if name is None:
            manager_id = 1 + self.nb_managers
            name = "manager_{}".format(manager_id)
        self.managers.update({name: manager})
        return

    def initialize_all(self):
        for manager in self.managers.itervalues():
            manager.initialize_all()
        return

    def start_all(self):
        for manager in self.managers.itervalues():
            manager.start_all()
        return

    def sleep(self, duration=None):
        self.log.debug("director sleeps {d} sec".format(d=duration))
        time.sleep(duration)
        return

    def stop_all(self):
        for manager in self.managers.itervalues():
            manager.stop_all()
        return

    def destroy_all(self):
        return
