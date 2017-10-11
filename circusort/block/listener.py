import numpy as np
import socket

from .block import Block


class Listener(Block):
    """Listener block"""
    # TODO complete docstring.

    name = "Stream listener"

    params = {
        'acq_host': '127.0.0.1',
        'acq_port': 40006,
        'acq_dtype': 'uint16',
        'acq_nb_samp': 2000,
        'acq_nb_chan': 261,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('data')

    def _initialize(self):
        # Configure the data output of this block.
        self.output.configure(dtype=self.acq_dtype, shape=(self.acq_nb_samp, self.acq_nb_chan))
        # Define the address of the input socket.
        address = (self.acq_host, self.acq_port)
        # Bind the input socket.
        self.acq_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.log.debug("Socket created.")
        # Connect to the server.
        # TODO remove following line.
        self.log.debug("{}".format(address))
        self.acq_socket.connect(address)
        self.log.debug("Connection accepted to {}:{}.".format(self.acq_host, self.acq_port))
        # Initialize counter for buffer receptions.
        self.step_nb = 0
        # Initialize buffer size.
        self.buf_size = self.acq_nb_chan * self.acq_nb_samp * 2
        return

    def _process(self):
        try:
            # Receive UDP packets.
            recv_string = self.acq_socket.recv(self.buf_size, socket.MSG_WAITALL)
            # Log reception.
            log_format = "{} len(recv_string): {}"
            log_string = log_format.format(self.step_nb, len(recv_string))
            self.log.debug(log_string)
            # Change data format.
            batch = self.read_live_udp_packet(recv_string, self.acq_dtype, self.acq_nb_chan)
            # Log data format.
            log_format = "{} batch.shape: {}"
            log_string = log_format.format(self.step_nb, batch.shape)
            self.log.debug(log_string)
            # Send output.
            self.output.send(batch)
            # Increment counter for buffer receptions.
            self.step_nb += 1
        except:
            raise NotImplementedError()
        return

    @staticmethod
    def read_live_udp_packet(acq_string, acq_dtype, acq_nb_chan):
        """"""
        # TODO add docstring.
        acq_shape = (-1, acq_nb_chan)
        acq_data = np.fromstring(acq_string, dtype=acq_dtype)
        acq_data = np.reshape(acq_data, acq_shape)
        return acq_data

    def __del__(self):
        # Close the input socket.
        self.acq_socket.close()
        return
