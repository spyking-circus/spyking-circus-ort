from multiprocessing import Process
from time import sleep
import zmq

from .receptacle import Receptacle



class ComputerProcess(Process):

    def __init__(self):
        super(ComputerProcess, self).__init__()

    def bind(self):
        print("Computer.bind...")
        self.context = zmq.Context()
        self.output = self.context.socket(zmq.PUSH)
        self.output.bind("tcp://*:4243")
        return

    def connect(self):
        print("Computer.connect...")
        self.context = zmq.Context()
        self.input = self.context.socket(zmq.PULL)
        self.input.connect("tcp://localhost:4242")
        return

    def begin(self):
        print("Computer.start...")
        i = 0
        while i < 10:
            data = self.input.recv()
            print("Computer compute data: ' {} '".format(data))
            self.output.send(data)
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



class Computer(object):

    def __init__(self):
        self.input = Receptacle()
        self.output = Receptacle()

    def configure(self):
        return

    def initialize(self):
        self.process = ComputerProcess()
        return

    def start(self):
        self.process.start()
        return

    def stop(self):
        self.process.terminate()
        self.process.join()
        return
