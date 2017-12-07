import tempfile
import os

from .block import Block


class Writer(Block):
    """Writer."""
    # TODO add docstring

    name = "File writer"

    params = {
        'data_path': None
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_input('data')

    @staticmethod
    def _get_temp_file():
        tmp_file = tempfile.NamedTemporaryFile()
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
            self.file.write(batch.tostring())
        else:
            self.log.error('{n} can only write arrays'.format(n=self.name))
        return

    def __del__(self):
        self.file.close()
