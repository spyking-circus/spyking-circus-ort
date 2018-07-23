import errno
import numpy as np
try:
    from Queue import Queue  # Python 2 compatibility.
except ImportError:  # i.e. ModuleNotFoundError
    from queue import Queue  # Python 3 compatibility.
import socket
import threading

from .block import Block


__classname__ = 'MCSReceiver'


class MCSReceiver(Block):

    name = "Mcs_receiver"

    params = {
        'dtype': 'uint16',
        'nb_channels': 261,
        'nb_samples': 2000,
        'sampling_rate': 20000, 
        'host': '127.0.0.1',
        'port': 8888,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        self.add_output('data')

        # The following lines are useful to disable PyCharm warnings.
        self.dtype = self.dtype
        self.nb_channels = self.nb_channels
        self.nb_samples = self.nb_samples
        self.sampling_rate = self.sampling_rate
        self.host = self.host
        self.port = self.port

    def _initialize(self):

        self.output.configure(dtype=self.dtype, shape=(self.nb_samples, self.nb_channels))

        self.queue = Queue()
        self.size = self.nb_channels * self.nb_samples * 2  # i.e. nb_chan * nb_step * size(uint16)

        def recv_target(queue, size, host, port):
            # Define the address of the input socket.
            address = (host, port)
            # Bind an input socket.
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Create a connection to this address.
            s.connect(address)
            # Receive data.
            while True:
                try:
                    recv_string = s.recv(size, socket.MSG_WAITALL)
                except socket.error as e:
                    if e.errno == errno.ECONNRESET:
                        # Discard error message.
                        break
                    else:
                        raise e
                queue.put(recv_string)

        # Prepare background thread for data acquisition.
        args = (self.queue, self.size, self.host, self.port)
        self.recv_thread = threading.Thread(target=recv_target, args=args)
        self.recv_thread.deamon = True

        # Launch background thread for data acquisition.
        self.log.info("{n} starts listening for data on {f}...".format(n=self.name, f="%s:%d" % (self.host, self.port)))
        self.recv_thread.start()

        return

    def _process(self):

        recv_string = self.queue.get()
        recv_shape = (-1, self.nb_channels)
        recv_data = np.fromstring(recv_string, dtype=self.dtype)
        recv_data = np.reshape(recv_data, recv_shape)

        self.output.send(recv_data)

        return
