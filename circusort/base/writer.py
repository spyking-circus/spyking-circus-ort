from multiprocessing import Process
from time import sleep
import zmq

from .receptacle import Receptacle



class WriterProcess(Process):

    def __init__(self, path):
        super(WriterProcess, self).__init__()
        self.path = path

    def bind(self):
        print("Writer.bind...")
        self.output = open(self.path, mode='wb')
        sleep(1.0)
        return

    def connect(self):
        print("Writer.connect...")
        self.context = zmq.Context()
        self.input = self.context.socket(zmq.PULL)
        self.input.connect("tcp://localhost:4243")
        sleep(1.0)
        return

    def begin(self):
        print("Writer.start...")
        i = 0
        while i < 10:
            data = self.input.recv()
            print("Writer write data: ' {} '".format(data))
            self.output.write(data)
            i += 1
            sleep(0.5)
        sleep(1.0)
        return

    # def end(self):
    #     return
    #
    # def disconnect(self):
    #     return
    #
    # def unbind(self):
    #     return

    def run(self):
        self.bind()
        self.connect()
        self.begin()
        # self.end()
        # self.disconnect()
        # self.unbind()
        return



class Writer(object):

    def __init__(self):
        self.input = Receptacle()

    def configure(self, path):
        if path is None:
            raise NotImplementedError()
        else:
            self.path = path
        return

    def initialize(self):
        self.process = WriterProcess(self.path)
        return

    def start(self):
        self.process.start()
        return

    def stop(self):
        self.process.terminate()
        self.process.join()
        return
