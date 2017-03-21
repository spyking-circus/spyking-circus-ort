from circusort.base import utils



class Writer(object):
    '''TODO add docstring'''

    def __init__(self, log_address=None):

        object.__init__(self)

        self.log_address = log_address

        self.name = "Writer's name (original)"
        if self.log_address is None:
            raise NotImplementedError("no logger address")
        self.log = utils.get_log(self.log_address, name=__name__)

    def initialize(self):
        '''TODO add docstring'''

        # TODO validate and implement this method

        # raise NotImplementedError()
        return
