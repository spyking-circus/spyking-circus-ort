import time

from .manager import Manager



class Director(object):

    def __init__(self):
        self.managers = {}

    @property
    def nb_managers(self):
        return len(self.managers)

    def create_manager(self, address=None):
        if address is None:
            manager = Manager()
        else:
            # TODO create manager proxy/client...
            raise NotImplementedError()
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
        time.sleep(duration)

    def stop_all(self):
        for manager in self.managers.itervalues():
            manager.stop_all()
        return

    def destroy_all(self):
        return
