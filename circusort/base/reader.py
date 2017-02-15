from multiprocessing import Process
import numpy
import os
from time import sleep
import zmq

from .receptacle import Receptacle



class ReaderProcess(Process):

    def __init__(self, path):
        super(ReaderProcess, self).__init__()
        self.path = path
        self.address = "tcp://*:4242"

    def bind(self):
        print("Reader.bind...")
        self.context = zmq.Context()
        self.output = self.context.socket(zmq.PUSH)
        self.output.bind(self.address)
        sleep(1.0)
        return

    def connect(self):
        print("Reader.connect...")
        # Create file if necessary
        if not os.path.isfile(self.path):
            size = 1000
            a = 256.0 * numpy.random.rand(size)
            a = a.astype('uint8')
            a.tofile(self.path)
        self.input = open(self.path, mode='rb')
        sleep(1.0)
        return

    def begin(self, i_max=10, buffer_size=4):
        print("Reader.start...")
        i = 0
        while i < i_max:
            data = self.input.read(buffer_size)
            print("Reader read data: ' {} '".format(data))
            self.output.send(data)
            i += 1
            sleep(0.5)
        # # or
        # while True:
        #     # 1. read chunk from file
        #     buffer_size = 1024
        #     data = self.file.read(buffer_size)
        #     print("data: {}".format(data))
        #     # 2. send chunk through the output
        #     self.socket.send(data)
        #     # 3. check if we have to stop
        #     #   a. check if manager have send a message
        #     #   b. if yes then stop
        #     pass
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


class Reader(object):

    def __init__(self):
        self.output = Receptacle()

    def configure(self, path=None):
        if path is None:
            # TODO use a default path...
            raise NotImplementedError()
        else:
            self.path = path
        return

    def initialize(self):
        self.process = ReaderProcess(self.path)
        return

    def start(self):
        self.process.start()
        return

    def stop(self):
        self.process.terminate()
        self.process.join()
        return
