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

        self.log.info("create block locally")

        process = Process(log_address=self.log_address)
        module = process.get_module('circusort.block.{n}'.format(n=name))
        block = getattr(module, name.capitalize())(log_address=self.log_address)

        return block
