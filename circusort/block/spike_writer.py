from .block import Block
import tempfile
import os
import numpy


class Spike_writer(Block):
    """Spike writer block.

    Attributes:
        spike_times: string
            Path to the location where spike times will be saved.
        amplitudes: string
            Path to the location where spike amplitudes will be saved.
        templates: string
            Path to the location where spike templates will be saved.
        directory: string
            Path to the location where spike attributes will be saved.

    """
    # TODO complete docstring.

    name = "Spike writer"

    params = {
        'spike_times': None,
        'amplitudes': None,
        'templates': None,
        'directory': None,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_input('spikes')

    def _get_temp_file(self, basename=None):

        if self.directory is None:
            tmp_dir = tempfile.gettempdir()
        else:
            tmp_dir = self.directory
        if basename is None:
            tmp_file = tempfile.NamedTemporaryFile()
            tmp_basename = os.path.basename(tmp_file.name)
            tmp_file.close()
        else:
            tmp_basename = basename
        tmp_filename = tmp_basename + ".raw"
        data_path = os.path.join(tmp_dir, tmp_filename)

        return data_path

    def _initialize(self):

        self.recorded_data = {}
        self.data_file = {}

        key = 'spike_times'
        if self.spike_times is None:
            self.recorded_data[key] = self._get_temp_file(basename='spike_times')
        else:
            self.recorded_data[key] = self.spike_times
        self.log.info('{n} records {m} into {k}'.format(n=self.name, m=key, k=self.recorded_data[key]))
        self.data_file[key] = open(self.recorded_data[key], mode='wb')
        key = 'amplitudes'
        if self.amplitudes is None:
            self.recorded_data[key] = self._get_temp_file(basename='amplitudes')
        else:
            self.recorded_data[key] = self.amplitudes
        self.log.info('{n} records {m} into {k}'.format(n=self.name, m=key, k=self.recorded_data[key]))
        self.data_file[key] = open(self.recorded_data[key], mode='wb')
        key = 'templates'
        if self.templates is None:
            self.recorded_data[key] = self._get_temp_file(basename='templates')
        else:
            self.recorded_data[key] = self.templates
        self.log.info('{n} records {m} into {k}'.format(n=self.name, m=key, k=self.recorded_data[key]))
        self.data_file[key] = open(self.recorded_data[key], mode='wb')

        return

    def _process(self):

        batch = self.input.receive()

        if self.input.structure == 'array':
            self.log.error('{n} can only write spike dictionaries'.format(n=self.name))
        elif self.input.structure == 'dict':
            offset = batch.pop('offset')
            for key in batch:
                # TODO remove the following commented lines.
                # if key not in self.recorded_data:
                #     if key == 'spike_times':
                #         if self.spike_times is None:
                #             self.recorded_data[key] = self._get_temp_file(basename='spike_times')
                #         else:
                #             self.recorded_data[key] = self.spike_times
                #     elif key == 'amplitudes':
                #         if self.amplitudes is None:
                #             self.recorded_data[key] = self._get_temp_file(basename='spike_amplitudes')
                #         else:
                #             self.recorded_data[key] = self.amplitudes
                #     elif key == 'templates':
                #         if self.templates is None:
                #             self.recorded_data[key] = self._get_temp_file(basename='spike_templates')
                #         else:
                #             self.recorded_data[key] = self.templates
                #
                #     self.log.info('{n} records {m} into {k}'.format(n=self.name, m=key, k=self.recorded_data[key]))
                #     self.data_file[key] = open(self.recorded_data[key], mode='wb')
                if key in ['spike_times']:
                    to_write = numpy.array(batch[key]).astype(numpy.int32)
                    to_write += offset
                elif key in ['templates']:
                    to_write = numpy.array(batch[key]).astype(numpy.int32)
                elif key in ['amplitudes']:
                    to_write = numpy.array(batch[key]).astype(numpy.float32)
                else:
                    raise KeyError(key)
                # TODO remove the following line.
                self.log.debug("{n} write in {k} file".format(n=self.name, k=key))
                self.data_file[key].write(to_write)
                self.data_file[key].flush()
        else:
            self.log.error("{n} can't write {s}".format(n=self.name, s=self.input.structure))

        return

    def __del__(self):

        for file in self.data_file.values():
            file.close()
