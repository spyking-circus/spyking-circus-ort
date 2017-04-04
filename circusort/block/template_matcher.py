from .block import Block


class Template_matcher(Block):
    '''TODO add docstring'''

    name   = "Template matcher"

    params = {'data_path'  : '/tmp/output.dat'}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_input('templates')
        self.add_input('data')
        self.add_input('peaks')

    def _initialize(self):
        '''TODO add docstring'''
        self.file = open(self.data_path, mode='wb')

        return

    def _process(self):
        batch = self.inputs['data'].receive()
        templates = self.inputs['templates'].receive()


        return
