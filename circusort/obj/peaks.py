import h5py
import numpy as np


class Peaks(object):

    def __init__(self, path):

        self._path = path
        self._file = h5py.File(self._path, mode='r')

    @property
    def _times(self):

        dataset = self._file['times']
        times = dataset.value

        return times

    @property
    def _nb_times(self):

        return self._times.size

    @property
    def _channels(self):

        dataset = self._file['channels']
        channels = dataset.value

        return channels

    @property
    def _polarities(self):

        dataset = self._file['polarities']
        polarities = dataset.value

        return polarities

    def get_times(self, t_min=None, t_max=None, channels=None, polarities=None):

        flags = np.ones(self._nb_times, dtype=np.bool)

        if t_min is not None:
            flags = np.logical_and(flags, t_min <= self._times)

        if t_max is not None:
            flags = np.logical_and(flags, self._times <= t_max)

        if channels is not None:
            flags_ = np.zeros(self._nb_times, dtype=np.bool)
            for channel in channels:
                flags_ = np.logical_or(flags_, self._channels == channel)
            flags = np.logical_and(flags, flags_)

        if polarities is not None:
            flags_ = np.zeros(self._nb_times, dtype=np.bool)
            for polarity in polarities:
                flags_ = np.logical_or(flags_, self._polarities == polarity)
            flags = np.logical_and(flags, flags_)

        times = self._times[flags]

        return times
