import numpy as np
import socket

from .block import Block


class Listener(Block):
    """Listener block"""
    # TODO complete docstring.

    name = "Stream listener"

    params = {
        'host': '127.0.0.1',
        'port': 4006,
        'dtype': 'uint16',
        'nb_chan': 261,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('data')

    def _initialize(self):
        # Define the address of the input socket.
        self.address = (self.host, self.port)
        # Bind the input socket.
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.log.debug("Socket created.")
        # Connect to the server.
        self.socket.connect(self.address)
        self.log.debug("Connection accepted to {}:{}.".format(self.host, self.port))
        # Initialize counter for receptions.
        self.step_nb = 0
        return

    def _process(self):
        try:
            # Receive UDP packets.
            recv_string = self.socket.recv(self.buf_size, socket.MSG_WAITALL)
            # Log reception.
            log_format = "{} len(recv_string): {}"
            log_string = log_format.format(self.step_nb, len(recv_string))
            self.log.debug(log_string)
            # Change data format.
            batch = self.read_live_udp_packet(recv_string, self.recv_dtype, self.nb_recv_chan)
            # Log data format.
            log_format = "{} batch.shape: {}"
            log_string = log_format.format(self.step_nb, batch.shape)
            self.log.debug(log_string)
            # Send output.
            self.output.send(batch)
        finally:
            raise NotImplementedError()
        return

    @staticmethod
    def read_live_udp_packet(recv_string, recv_dtype, nb_recv_chan):
        """"""
        # TODO add docstring.
        recv_shape = (-1, nb_recv_chan)
        recv_data = np.fromstring(recv_string, dtype=recv_dtype)
        recv_data = np.reshape(recv_data, recv_shape)
        return recv_data

    def __del__(self):
        # Close the input socket.
        self.socket.close()
        return
