import time

from .manager import Manager



class Director(object):

    def __init__(self):
        self.managers = {}

    @property
    def nb_managers(self):
        return len(self.managers)

    def create_manager(self, interface=None):
        '''Create a new manager process and return a proxy to this process.

        A manager is a process that manages workers.
        '''
        # TODO check if address is local or not...
        # 1. local -> create new process locally (if necessary)            [cli]
        # 2. network -> create new process remotely (if necessary)   [ssh + cli]

        # TODO remove code attempt...
        # process = Process(address=address)
        # module = process.client._import('circusort.base.manager')
        # manager = module.Manager(director=self)

        manager = Manager(interface=interface)
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
        print("Director start sleeping ({d} sec)...".format(d=duration))
        time.sleep(duration)
        return

    def stop_all(self):
        for manager in self.managers.itervalues():
            manager.stop_all()
        return

    def destroy_all(self):
        return
