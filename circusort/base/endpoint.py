class Endpoint(object):
    '''TODO add docstring'''

    def __init__(self, block):

        self.block = block

        self.dtype = None
        self.shape = None
        self.addr = None

        self.socket = None

    def __del__(self):

        if self.socket is not None:
            self.socket.close()

    def configure(self, dtype=None, shape=None, addr=None):
        '''TODO add docstring'''

        if dtype is not None:
            self.dtype = dtype

        if shape is not None:
            self.shape = shape

        if addr is not None:
            self.addr = addr

        return
