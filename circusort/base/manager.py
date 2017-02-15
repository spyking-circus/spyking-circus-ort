from time import sleep

from .reader import Reader
from .computer import Computer
from .writer import Writer



class Manager(object):

    def __init__(self):
        self.workers = {}

    @property
    def nb_workers(self):
        return len(self.workers)

    def create_reader(self):
        reader = Reader()
        self.register_worker(reader)
        return reader

    def create_computer(self):
        computer = Computer()
        self.register_worker(computer)
        return computer

    def create_writer(self):
        writer = Writer()
        self.register_worker(writer)
        return writer

    def register_worker(self, worker, name=None):
        if name is None:
            identifier = 1 + self.nb_workers
            name = "worker_{}".format(identifier)
        self.workers.update({name: worker})
        return

    def initialize_all(self):
        for worker in self.workers.itervalues():
            worker.initialize()
        return

    def start_all(self):
        for worker in self.workers.itervalues():
            worker.start()
        return

    def stop_all(self):
        for worker in self.workers.itervalues():
            worker.stop()
        return
