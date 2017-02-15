import multiprocessing
import zmq



class Connection(object):

    def bind(self):
        raise NotImplementedError()

    def connect(self):
        raise NotImplementedError()

    def send(self, data):
        raise NotImplementedError()

    def recv(self):
        raise NotImplementedError()

    def unbind(self):
        raise NotImplementedError()

    def disconnect(self):
        raise NotImplementedError()


class FileConnection(Connection):

    def __init__(self, path, buffer_size=1024):
        self.path = path
        self.buffer_size = buffer_size
        self.input_file = None

    def bind(self):
        self.output_file = open(self.path, 'a')
        return

    def connect(self):
        self.input_file = open(self.path, 'r')
        return

    def send(self, data):
        self.output_file.write(data)
        return

    def recv(self):
        data = self.input_file.read(self.buffer_size)
        return data

    def unbind(self):
        self.output_file.close()
        return

    def disconnect(self):
        self.input_file.close()
        return


class ZmqConnection(Connection):

    def __init__(self, address):
        self.address = address

    def bind(self):
        self.output_context = zmq.Context()
        self.output_socket = self.output_context.socket(zmq.PUSH)
        self.output_socket.bind(self.address)
        return

    def connect(self):
        self.input_context = zmq.Context()
        self.input_socket = self.input_context.socket(zmq.PULL)
        self.input_socket.connect(self.address)
        return

    def send(self, message):
        self.output_socket.send(message)
        return

    def recv(self):
        message = self.input_socket.recv()
        return message

    def unbind(self):
        self.output_socket.unbind(self.address)
        return

    def disconnect(self):
        self.input_socket.disconnect(self.address)
        return


class RPCConnection(Connection):

    def __init__(self):
        pass


class Process(object):

    def __init__(self, target, args=(), kwargs={}):
        self.target = target
        self.args = args
        self.kwargs = kwargs

    def initialize(self):
        multiprocessing.Process(target=self.target, args=self.args, kwargs=self.kwargs)
        raise NotImplementedError()

    def bind(self):
        raise NotImplementedError()

    def connect(self):
        raise NotImplementedError()

    def start(self):
        raise NotImplementedError()

    def stop(self):
        raise NotImplementedError()

    def disconnect(self):
        raise NotImplementedError()

    def unbind(self):
        raise NotImplementedError()

    def finalize(self):
        raise NotImplementedError()


def read(rpc_connection, input_connection=None, output_connection=None):
    rpc_connection.connect()
    while True:
        message = rpc_connection.recv()
        if message == "BIND":
            pass
        elif message == "CONNECT":
            pass
        elif message == "START":
            pass
    return


class Reader(Process):

    def __init__(self, input_connection, output_connection):
        self.input_connection = input_connection
        self.output_connection = output_connection
        self.rpc_connection = RPCConnection()

    def initialize(self):
        read_args = (self.rpc_connection,)
        read_kwargs = {
            'input_connection': self.input_connection,
            'output_connection': self.output_connection
        }
        self.process = Process(target=read, args=read_args, kwargs=read_kwargs)
        return

    def bind(self):
        self.rpc_connection.send("BIND")
        return

    def connect(self):
        self.rpc_connection.send("CONNECT")
        return

    def start(self):
        self.rpc_connection.send("START")

    # def start(self):
    #     self.input_connection.connect()
    #     self.output_connection.bind()
    #     while True:
    #         chunk = self.input_connection.recv()
    #         self.output_connection.send(chunk)
    #     self.output_connection.unbind()
    #     self.input_connection.disconnect()
    #     return

    def stop(self):
        self.process.terminate()

    def disconnect(self):
        self.rpc_connection.send("DISCONNECT")
        return

    def unbind(self):
        self.rpc_connection.send("UNBIND")
        return

    def finalize(self):
        self.rpc_connection.send("FINALIZE")


class Computer(Process):

    def __init__(self, input_connection, output_connection):
        self.input_connection = input_connection
        self.output_connection = output_connection

    def run(self):
        self.input_connection.connect()
        self.output_connection.bind()
        while True:
            chunk = self.input_connection.recv()
            self.output_connection.send(chunk)
        self.output_connection.unbind()
        self.input_connection.disconnect()
        return


class Writer(Process):

    def __init__(self, input_connection, output_connection):
        self.input_connection = input_connection
        self.output_connection = output_connection

    def run(self):
        self.input_connection.connect()
        self.output_connection.bind()
        while True:
            chunk = self.input_connection.recv()
            self.output_connection.send(chunk)
        self.output_connection.unbind()
        self.input_connection.disconnect()
        return


r_path = "" # input file path
w_path = "" # output file path

rc_address = "icp://0.0.0.0:4242:" # read/compute interface (i.e. address)
cw_address = "icp://0.0.0.0:4242:" # compute/write interface (i.e. address)

rc_connection = ZmqConnection(rc_address)
cw_connection = ZmqConnection(cw_address)

reader = Reader(r_path, rc_connection)
computer = Computer(rc_connection, cw_connection)
writer = Writer(cw_connection, w_path)

reader.initialize()
computer.initialize()
writer.initialize()

reader.bind()
computer.bind()
writer.bind()

reader.connect()
computer.connect()
writer.connect()

reader.start()
computer.start()
writer.start()

reader.stop()
computer.stop()
writer.stop()

reader.unbind()
computer.unbind()
writer.unbind()

reader.finalize()
computer.finalize()
writer.finalize()
