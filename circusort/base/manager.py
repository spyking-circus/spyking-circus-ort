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
        self.blocks_types = {}

        self.name = name or "Manager"
        if self.log_address is None:
            raise NotImplementedError("no logger address")
        self.log = utils.get_log(self.log_address, name=__name__, log_level=self.log_level)
        self.log.info("{d} starts".format(d=str(self)))

    def create_block(self, block_type, name=None, log_level=None, **kwargs):
        '''TODO add docstring'''

        if name is None:
            if self.blocks_types.has_key(block_type):
                suffix = 1 + self.blocks_types[block_type]
                self.blocks_types[block_type] += 1
            else:
                suffix = 1
                self.blocks_types[block_type] = 1
                

        if log_level is None:
            log_level = self.log_level

        process = Process(log_address=self.log_address, name="{n}".format(n=block_type), log_level=log_level)
        module = process.get_module('circusort.block.{n}'.format(n=block_type))
        block = getattr(module, block_type.capitalize())(log_address=self.log_address, log_level=log_level, **kwargs)

        if name is None:
            block.name = block.name + ' %d' %suffix
        else:
            block.name = name

        self.log.info("{d} creates block {s}[{n}]".format(d=str(self), s=block.name, n=block_type))
        
        self.register_block(block)

        return block

    def connect(self, output_endpoints, input_endpoints, protocol='ipc'):
        '''TODO add docstring'''

        if not type(input_endpoints) == list:
            input_endpoints = [input_endpoints]

        if not type(output_endpoints) == list:
            output_endpoints = [output_endpoints]

        for input_endpoint in input_endpoints:
            for output_endpoint in output_endpoints:
        
                self.log.info("{d} connects {s} to {t}".format(d=str(self), s=output_endpoint.block.name, t=input_endpoint.block.name))
                assert input_endpoint.block.parent == output_endpoint.block.parent == self.name, self.log.error('Manager is not supervising all Blocks!')
                assert protocol in ['tcp', 'ipc'], self.log.error('Invalid connection')

                output_endpoint.initialize(protocol=protocol, host=output_endpoint.block.host)
                description = output_endpoint.get_description()
                input_endpoint.configure(**description)
                input_endpoint.block.connect(input_endpoint.name)
                input_endpoint.block.guess_output_endpoints()
                self.log.debug("Connection established from {a}[{s}] to {b}[{t}]".format(s=(output_endpoint.name, output_endpoint.structure), 
                                                                                                    t=(input_endpoint.name, input_endpoint.structure), 
                                                                                                    a=output_endpoint.block.name,
                                                                                                    b=input_endpoint.block.name))


        return

    def __str__(self):
        return "{d}[{i}]".format(d=self.name, i=self.host)

    @property
    def nb_blocks(self):
        return len(self.blocks)

    def register_block(self, block):
        block.set_manager(self.name)
        block.set_host(self.host)
        assert block.name not in self.blocks.keys(), self.log.error('Two blocks with the same name {n}'.format(n=block.name))
        self.blocks.update({block.name: block})
        self.log.debug("{d} registers {m}".format(d=str(self), m=block.name))
        return

    def list_blocks(self):
        return self.blocks.keys()

    def get_block(self, key):
        assert key in self.list_blocks(), self.log.error("%s is not a valid block" %key)
        return self.blocks[key]

    def initialize(self):
        self.log.info("{d} initializes {s}".format(d=str(self), s=", ".join(self.list_blocks())))
        for block in self.blocks.itervalues():
            block.initialize()
        return

    def join(self):
        self.log.info("{d} joins {s}".format(d=str(self), s=", ".join(self.list_blocks())))
        for block in self.blocks.itervalues():
            block.join()
        return

    def start(self, nb_steps=None):
        if nb_steps is None:
            self.log.info("{d} starts {s}".format(d=str(self), s=", ".join(self.list_blocks())))
            for block in self.blocks.itervalues():
                block.start()
        else:
            self.log.info("{d} runs {s} for {n} steps".format(d=str(self), s=", ".join(self.list_blocks()), n=nb_steps))
            for block in self.blocks.itervalues():
                block.nb_steps = nb_steps
                block.start()
                block.nb_steps = None
        return

    def stop(self):
        self.log.info("{d} stops {s}".format(d=str(self), s=", ".join(self.list_blocks())))
        for block in self.blocks.itervalues():
            block.stop()
        return