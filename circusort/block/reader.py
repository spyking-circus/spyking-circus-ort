from circusort.base import utils



class Reader(object):
    '''TODO add docstring'''

    def __init__(self, log_address=None):

        object.__init__(self)

        self.log_address = log_address

        self.name = "Reader's name (original)"
        if self.log_address is None:
            raise NotImplementedError("no logger address")
        self.log = utils.get_log(self.log_address, name=__name__)

        self.path = "/tmp/input.dat"
        self.dtype = 'float32'
        self.n_electrodes = 4

    def initialize(self):
        '''TODO add docstring'''

        # TODO validate and implement this method

        # raise NotImplementedError()
        return

    def connect(self):
        '''TODO add docstring'''

        # TODO validate and implement this method

        raise NotImplementedError()

    def start(self):
        '''TODO add dosctring'''

        # TODO validate and implement this method

        raise NotImplementedError()
