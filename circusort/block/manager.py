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
        self.blocks = {}

        self.name = name or "Manager"
        if self.log_address is None:
            raise NotImplementedError("no logger address")
        self.log = utils.get_log(self.log_address, name=__name__, log_level=self.log_level)
        self.log.info("start manager {d}".format(d=str(self)))

    def create_block(self, name, log_level=None, **kwargs):
        '''TODO add docstring'''

        self.log.info("{d} creates block {n}".format(d=str(self), n=name))
        if log_level is None:
            log_level = self.log_level

        process = Process(log_address=self.log_address, name="{n}".format(n=name), log_level=log_level)
        module = process.get_module('circusort.block.{n}'.format(n=name))
        block = getattr(module, name.capitalize())(log_address=self.log_address, log_level=log_level, **kwargs)

        self.register_block(block)

        return block

    def connect(self, input_endpoint, output_endpoint, protocol='tcp'):
        '''TODO add docstring'''

        self.log.info("{d} connects couple of blocks".format(d=str(self)))

        assert protocol in ['tcp', 'udp', 'ipc'], self.log.error('Invalid connection')

        input_endpoint.initialize(protocol=protocol)
        output_endpoint.initialize(protocol=protocol)

        input_endpoint.configure(addr=output_endpoint.addr)
        output_endpoint.configure(dtype=input_endpoint.dtype,
                                  shape=input_endpoint.shape)

        input_endpoint.block.connect()
        output_endpoint.block.connect()

        return

    def __str__(self):
        return "{d}[{i}]".format(d=self.name, i=self.host)

    @property
    def nb_blocks(self):
        return len(self.blocks)

    def register_block(self, block):
        
        self.blocks.update({block.name: block})
        self.log.debug("{d} registers {m}".format(d=str(self), m=block.name))
        return

    def list_blocks(self):
        return self.blocks.keys()

    def get_block(self, key):
        assert key in self.list_blocks(), self.log.error("%s is not a valid block" %key)
        return self.blocks[key]

    def initialize(self):
        for block in self.blocks.itervalues():
            block.initialize()
        return

    def join(self):
        for block in self.blocks.itervalues():
            block.join()
        return

    def start(self):
        for block in self.blocks.itervalues():
            block.start()
        return