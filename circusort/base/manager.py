import logging

from circusort.base.utils import get_log
from circusort.base.process import Process
from circusort import net as network_module


class Manager(object):
    """Manager"""
    # TODO complete docstring.

    def __init__(self, name=None, host=None, log_address=None, log_level=logging.INFO):

        object.__init__(self)

        self.log_address = log_address
        self.log_level = log_level
        self.host = host
        self.blocks = {}
        self.blocks_types = {}
        self._networks = {}
        self._networks_types = {}

        self.name = name or "Manager"
        if self.log_address is None:
            raise NotImplementedError("no logger address")
        self.log = get_log(self.log_address, name=__name__, log_level=self.log_level)

        string = "{} is created."
        message = string.format(str(self))
        self.log.info(message)

    def create_block(self, block_type, name=None, log_level=None, **kwargs):
        """Create a new block linked to the manager."""
        # TODO complete docstring.

        if name is None:
            if block_type in self.blocks_types:
                self.blocks_types[block_type] += 1
            else:
                self.blocks_types[block_type] = 1
            suffix = self.blocks_types[block_type]
        else:
            suffix = None

        if log_level is None:
            log_level = self.log_level

        process = Process(log_address=self.log_address, name="{n}".format(n=block_type), log_level=self.log_level)
        module = process.get_module('circusort.block.{n}'.format(n=block_type))
        block = getattr(module, block_type.capitalize())(log_address=self.log_address, log_level=log_level, **kwargs)

        if name is None:
            block.name = "{} {}".format(block.name, suffix)
        else:
            block.name = name

        string = "{} creates block {}[{}]"
        message = string.format(str(self), block.name, block_type)
        self.log.info(message)

        self.register_block(block)

        return block

    def create_network(self, network_type, name=None, log_level=None, **kwargs):
        """Create a new network linked to the manager."""
        # TODO complete docstring.

        if name is None:
            if network_type in self._networks_types:
                self._networks_types[network_type] += 1
            else:
                self._networks_types[network_type] = 1
            suffix = self._networks_types[network_type]
        else:
            suffix = None

        if log_level is None:
            log_level = self.log_level

        module = getattr(network_module, network_type)
        class_ = getattr(module, network_type.capitalize())
        network = class_(self, name=name, log_address=self.log_address, log_level=log_level, **kwargs)

        if name is None:
            network.name = "{} {}".format(network.name, suffix)
        else:
            network.name = name

        string = "{} creates network {}[{}]"
        message = string.format(str(self), network.name, network_type)
        self.log.info(message)

        return network

    def connect(self, output_endpoints, input_endpoints, protocol='ipc', show_log=True):
        """Connect endpoints"""
        # TODO complete docstring.

        if not type(input_endpoints) == list:
            input_endpoints = [input_endpoints]

        if not type(output_endpoints) == list:
            output_endpoints = [output_endpoints]

        for input_endpoint in input_endpoints:
            for output_endpoint in output_endpoints:
                if show_log:
                    string = "{} connects {} to {}."
                    message = string.format(str(self), output_endpoint.block.name, input_endpoint.block.name)
                    self.log.info(message)
                else:
                    string = "{} connects {} to {}."
                    message = string.format(str(self), output_endpoint.block.name, input_endpoint.block.name)
                    self.log.debug(message)
                message = "Manager is not supervising all blocks!"
                assert input_endpoint.block.parent == output_endpoint.block.parent \
                    and output_endpoint.block.parent == self.name, self.log.error(message)
                message = "Invalid connection."
                assert protocol in ['tcp', 'ipc'], self.log.error(message)

                output_endpoint.initialize(protocol=protocol, host=output_endpoint.block.host)
                description = output_endpoint.get_description()
                input_endpoint.configure(**description)
                input_endpoint.block.connect(input_endpoint.name)
                input_endpoint.block.guess_output_endpoints()

                string = "{p} connection established from {a}[{s}] to {b}[{t}]"
                message = string.format(
                    p=protocol,
                    s=(output_endpoint.name, output_endpoint.structure),
                    t=(input_endpoint.name, input_endpoint.structure),
                    a=output_endpoint.block.name,
                    b=input_endpoint.block.name
                )
                self.log.debug(message)

        return

    def __str__(self):

        return "{d}[{i}]".format(d=self.name, i=self.host)

    @property
    def nb_blocks(self):

        return len(self.blocks)

    def register_block(self, block):

        block.set_manager(self.name)
        block.set_host(self.host)

        string = "Two blocks with the same name {}"
        message = string.format(block.name)
        assert block.name not in self.blocks.keys(), self.log.error(message)

        self.blocks.update({block.name: block})

        string = "{} registers {}."
        message = string.format(str(self), block.name)
        self.log.debug(message)

        return

    def list_blocks(self):

        return self.blocks.keys()

    def get_block(self, key):

        string = "{} is not a valid block."
        message = string.format(key)
        assert key in self.list_blocks(), self.log.error(message)

        return self.blocks[key]

    def initialize(self):

        string = "{} initializes {}."
        message = string.format(str(self), ", ".join(self.list_blocks()))
        self.log.info(message)

        for block in self.blocks.itervalues():
            block.initialize()

        return

    def join(self):

        string = "{} joins {}."
        message = string.format(str(self), ", ".join(self.list_blocks()))
        self.log.debug(message)

        for block in self.blocks.itervalues():
            block.join()

        self.log.info(message)

        return

    @property
    def data_producers(self):

        res = []
        for block in self.blocks.itervalues():
            if block.nb_inputs == 0:
                res += [block]

        return res

    @property
    def data_consumers(self):

        res = []
        for block in self.blocks.itervalues():
            if block.nb_outputs == 0:
                res += [block]

        return res

    @property
    def data_consumers_and_producers(self):

        res = []
        for block in self.blocks.itervalues():
            if block.nb_inputs != 0 and block.nb_outputs != 0:
                res += [block]

        return res

    def start(self, nb_steps=None):

        if nb_steps is None:

            string = "{} starts {}"
            message = string.format(str(self), ", ".join(self.list_blocks()))
            self.log.info(message)

            for sources in [self.data_consumers, self.data_consumers_and_producers, self.data_producers]:
                for block in sources:
                    block.start()

        else:

            string = "{} runs {} for {} steps"
            message = string.format(str(self), ", ".join(self.list_blocks()), nb_steps)
            self.log.info(message)

            for sources in [self.data_consumers, self.data_consumers_and_producers, self.data_producers]:
                for block in sources:
                    block.nb_steps = nb_steps
                    block.start()
                    block.nb_steps = None

        return

    def stop(self):

        string = "{} stops {}."
        message = string.format(str(self), ", ".join(self.list_blocks()))
        self.log.debug(message)

        for block in self.blocks.itervalues():
            block.stop()

        self.log.info(message)

        return

    def __del__(self):

        for block in self.blocks.itervalues():
            block.__del__()

        string = "{} is destroyed."
        message = string.format(str(self))
        self.log.info(message)
