from circusort.base import utils
from circusort.base.process import Process


class Manager(object):
    '''TODO add docstring'''

    def __init__(self, log_address=None):

        object.__init__(self)

        self.log_address = log_address

        self.name = "Manager's name (original)"
        if self.log_address is None:
            raise NotImplementedError("no logger address")
        self.log = utils.get_log(self.log_address, name=__name__)

    def create_block(self, name):
        '''TODO add docstring'''

        self.log.info("create {n} block".format(n=name))

        process = Process(log_address=self.log_address, name="{n}'s client".format(n=name))
        module = process.get_module('circusort.block.{n}'.format(n=name))
        block = getattr(module, name.capitalize())(log_address=self.log_address)

        return block

    def connect(self, input_endpoint, output_endpoint):
        '''TODO add docstring'''

        self.log.info("connect couple of blocks")

        input_endpoint.configure(addr=output_endpoint.addr)
        output_endpoint.configure(dtype=input_endpoint.dtype,
                                  shape=input_endpoint.shape)

        return
