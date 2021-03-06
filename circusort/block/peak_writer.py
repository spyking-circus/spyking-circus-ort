import h5py
import numpy as np
import os
import tempfile

from circusort.block.block import Block


__classname__ = 'PeakWriter'


class PeakWriter(Block):
    """Peak writer block

    Attributes:
        pos_peaks: string
            Path to the location to save positive peaks.
        neg_peaks: string
            Path to the location to save negative peaks.
        sampling_rate: float
            The sampling rate to use to transform timestamps into times.


    Input:
        peaks

    """

    name = "Peak writer"

    params = {
        'pos_peaks': None,
        'neg_peaks': None,
        'data_path': None,
        'sampling_rate': 20e+3,  # Hz
        'nb_samples': 1024,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_input('peaks', structure='dict')

        # The following lines are useful to avoid some PyCharm warnings.
        self.pos_peaks = self.pos_peaks
        self.neg_peaks = self.neg_peaks
        self.data_path = self.data_path
        self.sampling_rate = self.sampling_rate
        self.nb_samples = self.nb_samples

        if self.pos_peaks is not None or self.neg_peaks is not None:
            self._mode = 'raw'
        elif self.data_path is not None:
            self._mode = 'hdf5'
        else:
            self._mode = 'raw'  # TODO complete.

        self._h5_file = None

    @staticmethod
    def _get_temp_file():

        directory = tempfile.gettempdir()
        with tempfile.NamedTemporaryFile() as file_:
            name = os.path.basename(file_.name)
        filename = name + ".raw"
        path = os.path.join(directory, filename)

        return path

    def _initialize(self):

        self.recorded_peaks = {}
        self.peaks_file = {}

        if self._mode == 'raw':
            # Open file for positive peaks.
            if self.pos_peaks is None:
                self.recorded_peaks['positive'] = self._get_temp_file()
            else:
                self.recorded_peaks['positive'] = self.pos_peaks
            self.peaks_file['positive'] = open(self.recorded_peaks['positive'], mode='wb')
            # Open file for negative peaks.
            if self.neg_peaks is None:
                self.recorded_peaks['negative'] = self._get_temp_file()
            else:
                self.recorded_peaks['negative'] = self.neg_peaks
            self.peaks_file['negative'] = open(self.recorded_peaks['negative'], mode='wb')
        elif self._mode == 'hdf5':
            self._h5_file = h5py.File(self.data_path, mode='w', swmr=True)

        # Log info message.
        if self._mode == 'raw':
            string = "{} records data into {} (negative peaks) and {} (positive peaks)"
            message = string.format(self.name, self.recorded_peaks['negative'], self.recorded_peaks['positive'])
            self.log.info(message)
        elif self._mode == 'hdf5':
            # Log info message.
            string = "{} records data into {}"
            message = string.format(self.name, self.data_path)
            self.log.info(message)

        return

    def _process(self):

        # Receive input data.
        peaks_packet = self.get_input('peaks').receive()
        batch = peaks_packet['payload']['peaks']
        offset = peaks_packet['payload']['offset']

        self._measure_time('start')

        if self.input.structure == 'dict':

            if self._mode == 'raw':
                for key in batch:
                    to_write = []
                    for channel in batch[key].keys():
                        to_write += [(int(channel), value + offset) for value in batch[key][channel]]
                    to_write = np.array(to_write).astype(np.int32)
                    self.peaks_file[key].write(to_write)
                    self.peaks_file[key].flush()
            elif self._mode == 'hdf5':
                times = []
                channels = []
                polarities = []
                for key in batch:
                    if key == 'positive':
                        polarity = +1
                    elif key == 'negative':
                        polarity = -1
                    else:
                        polarity = 0
                    for channel in batch[key].keys():
                        times += [float(value + offset) / self.sampling_rate for value in batch[key][channel]]
                        channels += [int(channel) for _ in batch[key][channel]]
                        polarities += [polarity for _ in batch[key][channel]]
                data = {
                    'times': np.array(times, dtype=np.float32),
                    'channels': np.array(channels, dtype=np.uint16),
                    'polarities': np.array(polarities, dtype=np.int16),
                }
                for key in data:
                    if key in self._h5_file:
                        dataset = self._h5_file[key]
                        shape = dataset.shape
                        shape_ = (shape[0] + data[key].shape[0],)
                        dataset.resize(shape_)
                        dataset[shape[0]:] = data[key]
                        dataset.flush()
                    else:
                        dataset = self._h5_file.create_dataset(key, data=data[key], chunks=True, maxshape=(None,))
                        dataset.flush()
            else:
                # Raise value error.
                string = "Unknown mode value: {}"
                message = string.format(self._mode)
                raise ValueError(message)

        else:

            # Log error message.
            string = "{} can only write peak dictionaries"
            message = string.format(self.name)
            self.log.error(message)

        self._measure_time('end')

        return

    def _introspect(self):

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
            for file_ in self.peaks_file.values():
                file_.close()
        elif self._mode == 'hdf5':
            if self._h5_file is not None:
                self._h5_file.close()
