from .block import Block
import tempfile
import os
import numpy

class Spike_writer(Block):
    '''TODO add docstring'''

    name   = "Spike writer"

    params = {'spike_times' : None, 
              'amplitudes'  : None,
              'templates'   : None}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_input('spikes')

    def _get_temp_file(self):
        tmp_file  = tempfile.NamedTemporaryFile()
        data_path = os.path.join(tempfile.gettempdir(), os.path.basename(tmp_file.name)) + ".dat"
        tmp_file.close()
        return data_path

    def _initialize(self):
        self.recorded_data = {}
        self.data_file     = {}
        return

    def _process(self):
        batch = self.input.receive()
        if self.input.structure == 'array':
            self.log.error('{n} can only write spike dictionnaries'.format(n=self.name))
        elif self.input.structure == 'dict':
            offset = batch.pop('offset')
            for key in batch:
                if not self.recorded_data.has_key(key):
                    if key == 'spike_times':
                        if self.spike_times is None:
                            self.recorded_data[key] = self._get_temp_file()
                        else:
                            self.recorded_data[key] = self.spike_times
                    elif key == 'amplitudes':
                        if self.amplitudes is None:
                            self.recorded_data[key] = self._get_temp_file()
                        else:
                            self.recorded_data[key] = self.amplitudes
                    elif key == 'templates':
                        if self.templates is None:
                            self.recorded_data[key] = self._get_temp_file()
                        else:
                            self.recorded_data[key] = self.templates

                    self.log.info('{n} is recording {m} into {k}'.format(n=self.name, m=key, k=self.recorded_data[key]))
                    self.data_file[key] = open(self.recorded_data[key], mode='wb')
                
                if key in ['spike_times']:            
                    to_write = numpy.array(batch[key]).astype(numpy.int32)
                    to_write += offset
                elif key in ['templates']:
                    to_write = numpy.array(batch[key]).astype(numpy.int32)
                elif key in ['amplitudes']:
                    to_write = numpy.array(batch[key]).astype(numpy.float32)
                self.data_file[key].write(to_write)
                self.data_file[key].flush()
                os.fsync(self.data_file[key].fileno())
        return
