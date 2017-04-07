from .block import Block
import tempfile
import os
import numpy

class Peak_writer(Block):
    '''TODO add docstring'''

    name   = "Peak writer"

    params = {'pos_peaks' : None, 
              'neg_peaks' : None}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_input('peaks')

    def _get_temp_file(self):
        tmp_file  = tempfile.NamedTemporaryFile()
        data_path = os.path.join(tempfile.gettempdir(), os.path.basename(tmp_file.name)) + ".dat"
        tmp_file.close()
        return data_path

    def _initialize(self):
        self.recorded_peaks = {}
        self.peaks_file     = {}
        return

    def _process(self):
        batch = self.input.receive()
        if self.input.structure == 'array':
            self.log.error('{n} can only write peak dictionnaries'.format(n=self.name))
        elif self.input.structure == 'dict':
            offset = batch.pop('offset')
            for key in batch:
                if not self.recorded_peaks.has_key(key):
                    if key == 'positive':
                        if self.pos_peaks is None:
                            self.recorded_peaks[key] = self._get_temp_file()
                        else:
                            self.recorded_peaks[key] = self.pos_peaks
                    elif key == 'negative':
                        if self.neg_peaks is None:
                            self.recorded_peaks[key] = self._get_temp_file()
                        else:
                            self.recorded_peaks[key] = self.neg_peaks
                    self.log.info('{n} is recording {m} peaks into {k}'.format(n=self.name, m=key, k=self.recorded_peaks[key]))
                    self.peaks_file[key] = open(self.recorded_peaks[key], mode='wb')
                
                to_write = []
                for channel in batch[key].keys():
                    to_write += [(int(channel), value + offset) for value in batch[key][channel]]
                to_write = numpy.array(to_write).astype(numpy.int32)
                self.peaks_file[key].write(to_write)
        return
