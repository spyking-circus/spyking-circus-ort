from .block import Block
import tempfile
import os

class Writer(Block):
    '''TODO add docstring'''

    name   = "File writer"

    params = {'data_path'  : None}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_input('data')

    def _get_temp_file(self):
        tmp_file  = tempfile.NamedTemporaryFile()
        data_path = os.path.join(tempfile.gettempdir(), os.path.basename(tmp_file.name)) + ".dat"
        tmp_file.close()
        return data_path

    def _initialize(self):
        if self.data_path is None:
            self.data_path = self._get_temp_file()
        self.log.info('{n} records data into {k}'.format(n=self.name, k=self.data_path))
        self.file = open(self.data_path, mode='wb')
        self.recorded_keys = {}
        return

    def _process(self):
        batch = self.input.receive()
        if self.input.structure == 'array':
            try:
                self.file.write(batch.T.tostring())
            except ValueError as e:
                self.log.error("ValueError: {}".format(e.message))
        else:
            self.log.error('{n} can only write arrays'.format(n=self.name))
        return

    def __del__(self):
        self.file.close()
