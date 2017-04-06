from .block import Block
import tempfile


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
        self.log.info('{n} is recording into {k}'.format(n=self.name, k=self.data_path))
        self.file = open(self.data_path, mode='wb')
        self.recorded_keys = {}
        return

    def _process(self):
        batch = self.input.receive()
        if self.input.structure == 'array':
            self.file.write(batch.T.tostring())
        elif self.input.structure == 'dict':
            pass
            # for key in batch:
            #     if not self.recorded_keys.has_key(key):
            #         self.recorded_keys[key] = self._get_temp_file()
            #         self.log.info('{n} is recording key {m} into {k}'.format(n=self.name, m=key, k=self.recorded_keys[key]))
            #         self.recorded_keys[key] = open(self.recorded_keys[key], mode='wb')
            #     else:
            #         self.recorded_keys[key].write(batch[key])
        return
