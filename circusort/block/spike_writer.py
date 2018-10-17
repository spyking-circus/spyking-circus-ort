import h5py
import numpy as np
import os
import tempfile
import time

from circusort.block.block import Block


__classname__ = "SpikeWriter"


class SpikeWriter(Block):
    """Spike writer block.

    Attributes:
        data_path: string
            The path to the HDF5 file to use to write the data.
        spike_times: string
            Path to the location where spike times will be saved.
        amplitudes: string
            Path to the location where spike amplitudes will be saved.
        templates: string
            Path to the location where spike templates will be saved.
        rejected_times: string
            Path to the location where rejected times will be saved.
        rejected_amplitudes: string
            Path to the location where rejected amplitudes will be saved.
        directory: string
            Path to the location where spike attributes will be saved.
        sampling_rate: float
            The sampling rate to use to convert timestamps into times.
    """

    name = "Spike writer"

    params = {
        'data_path': None,
        'spike_times': None,
        'amplitudes': None,
        'templates': None,
        'rejected_times': None,
        'rejected_amplitudes': None,
        'directory': None,
        'sampling_rate': 20e+3,  # Hz
        'nb_samples': 1024,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        # The following lines are useful to avoid some PyCharm's warning.
        self.data_path = self.data_path
        self.spike_times = self.spike_times
        self.amplitudes = self.amplitudes
        self.templates = self.templates
        self.rejected_times = self.rejected_times
        self.rejected_amplitudes = self.rejected_amplitudes
        self.directory = self.directory
        self.sampling_rate = self.sampling_rate
        self.nb_samples = self.nb_samples

        self.add_input('spikes', structure='dict')

        if self.data_path is None:
            self._mode = 'raw'
        else:
            self._mode = 'hdf5'
        self._h5_file = None

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

        if self._mode == 'raw':
            self.recorded_data = {}
            self.data_file = {}
            self._initialize_data_file('spike_times', self.spike_times)
            self._initialize_data_file('amplitudes', self.amplitudes)
            self._initialize_data_file('templates', self.templates)
            self._initialize_data_file('rejected_times', self.rejected_times)
            self._initialize_data_file('rejected_amplitudes', self.rejected_amplitudes)
        elif self._mode == 'hdf5':
            self._h5_file = h5py.File(self.data_path, mode='w', swmr=True)

        return

    def _initialize_data_file(self, key, path):

        # Define path.
        if path is None:
            self.recorded_data[key] = self._get_temp_file(basename=key)
        else:
            self.recorded_data[key] = path
        # Log info message.
        string = "{} records {} into {}"
        message = string.format(self.name, key, self.recorded_data[key])
        self.log.info(message)
        # Open file.
        self.data_file[key] = open(self.recorded_data[key], mode='wb')

        return

    def _process(self):

        # Receive input spikes.
        spikes_packet = self.get_input('spikes').receive(blocking=False)
        batch = spikes_packet['payload'] if spikes_packet is not None else None

        if batch is None:

            duration = 0.1  # s
            time.sleep(duration)

        else:

            self._measure_time('start', frequency=100)

            if self.input.structure == 'dict':

                offset = batch.pop('offset')
                if self._mode == 'raw':
                    for key in batch:
                        if key in ['spike_times']:
                            to_write = np.array(batch[key]).astype(np.int32)
                            to_write += offset
                        elif key in ['templates']:
                            to_write = np.array(batch[key]).astype(np.int32)
                        elif key in ['amplitudes']:
                            to_write = np.array(batch[key]).astype(np.float32)
                        elif key in ['rejected_times']:
                            to_write = np.array(batch[key]).astype(np.int32)
                            to_write += offset
                        elif key in ['rejected_amplitudes']:
                            to_write = np.array(batch[key]).astype(np.float32)
                        else:
                            raise KeyError(key)
                        self.data_file[key].write(to_write)
                        self.data_file[key].flush()
                elif self._mode == 'hdf5':
                    for key in batch:
                        dataset_name = key
                        if key == 'spike_times':
                            dataset_name = 'times'
                            times = [
                                float(timestamp + offset) / self.sampling_rate
                                for timestamp in batch[key]
                            ]
                            data = np.array(times, dtype=np.float32)
                        elif key == 'templates':
                            data = np.array(batch[key], dtype=np.int16)
                        elif key == 'amplitudes':
                            data = np.array(batch[key], dtype=np.float32)
                        elif key == 'rejected_times':
                            rejected_times = [
                                float(timestamp + offset) / self.sampling_rate
                                for timestamp in batch[key]
                            ]
                            data = np.array(rejected_times, dtype=np.float32)
                        elif key == 'rejected_amplitudes':
                            data = np.array(batch[key], dtype=np.float32)
                        else:
                            data = None
                        if data is None or data.size == 0:
                            pass
                        elif dataset_name in self._h5_file:
                            dataset = self._h5_file[dataset_name]
                            shape = dataset.shape
                            shape_ = (shape[0] + data.shape[0],)
                            dataset.resize(shape_)
                            dataset[shape[0]:] = data
                            dataset.flush()
                        else:
                            dataset = self._h5_file.create_dataset(dataset_name, data=data,
                                                                   chunks=True, maxshape=(None,))
                            dataset.flush()

            else:

                # Log error message.
                string = "{} can only write spike dictionaries"
                message = string.format(self.name)
                self.log.error(message)

            self._measure_time('end', frequency=100)

        return

    def _introspect(self):
        """Introspection of this block for spike writing."""

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self.nb_samples) / self.sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return

    def __del__(self):

        if self._mode == 'raw':
            for file_ in self.data_file.values():
                file_.close()
        elif self._mode == 'hdf5':
            if self._h5_file is not None:
                self._h5_file.close()
