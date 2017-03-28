from circusort.base import utils
from circusort.base.process import Process
import logging

class Manager(object):
    '''TODO add docstring'''

    def __init__(self, name=None, host=None, log_address=None, log_level=logging.INFO):

        object.__init__(self)

        self.log_address = log_address
        self.log_level = log_level
        self.host = host

        self.name = name or "Manager"
        if self.log_address is None:
            raise NotImplementedError("no logger address")
        self.log = utils.get_log(self.log_address, name=__name__, log_level=self.log_level)
        self.log.info("start manager {d}".format(d=str(self)))

    def create_block(self, name, log_level=None):
        '''TODO add docstring'''

        self.log.info("{d} creates block {n}".format(d=str(self), n=name))
        if log_level is None:
            log_level = self.log_level

        process = Process(log_address=self.log_address, name="{n}".format(n=name), log_level=log_level)
        module = process.get_module('circusort.block.{n}'.format(n=name))
        block = getattr(module, name.capitalize())(log_address=self.log_address)

        return block

    def connect(self, input_endpoint, output_endpoint):
        '''TODO add docstring'''

        self.log.info("{d} connects couple of blocks".format(d=str(self)))

        input_endpoint.configure(addr=output_endpoint.addr)
        output_endpoint.configure(dtype=input_endpoint.dtype,
                                  shape=input_endpoint.shape)

        return

    # def list_blocks(self):
    #     return list_modules()

    def __str__(self):
        return "{d}[{i}]".format(d=self.name, i=self.host)