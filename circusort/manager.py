class Manager():
    '''TODO add docstring...'''
    def __init__(self, address=None, port=None):
        self.address = address
        self.port = port

    def __repr__(self):
        formatter = "manager (address: {}, port: {})"
        return formatter.format(self.address, self.port)
