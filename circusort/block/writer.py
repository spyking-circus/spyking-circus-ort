from .block import Block


class Writer(Block):
    '''TODO add docstring'''

    name   = "File writer"

    params = {'data_path'  : '/tmp/output.dat'}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_input('data')

    def _initialize(self):
        '''TODO add docstring'''
        self.file = open(self.data_path, mode='wb')

        return

    def _process(self):
        batch = self.input.receive()
        batch = batch.tobytes()
        self.file.write(batch)

        return
