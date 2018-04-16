import numpy as np


class MADFile(object):
    # TODO add docstring.

    def __init__(self, path, dtype, nb_channels, nb_samples_per_buffer, sampling_rate):
        # TODO add docstring.

        self._path = path
        self._dtype = dtype
        self._nb_channels = nb_channels
        self._nb_samples_per_buffer = nb_samples_per_buffer
        self._sampling_rate = sampling_rate

        self._data = np.memmap(self._path, dtype=self._dtype)

        assert self._data.size % self._nb_channels == 0
        self._nb_buffers = self._data.size // self._nb_channels
        self._data = self._data.reshape(self._nb_buffers, self._nb_channels)

    def get_snippet(self, t_min, t_max):
        # TODO add docstring.

        i_min = int(t_min * self._sampling_rate / float(self._nb_samples_per_buffer))
        i_max = int(t_max * self._sampling_rate / float(self._nb_samples_per_buffer)) + 1
        snippet = self._data[i_min:i_max, :]

        return snippet

    def get_snippet_times(self, t_min, t_max):
        # TODO add docstring.

        i_min = int(t_min * self._sampling_rate / float(self._nb_samples_per_buffer))
        i_max = int(t_max * self._sampling_rate / float(self._nb_samples_per_buffer)) + 1

        t_inf = float(i_min * self._nb_samples_per_buffer) / self._sampling_rate
        t_sup = float(i_max * self._nb_samples_per_buffer) / self._sampling_rate
        nb_points = i_max - i_min

        times = np.linspace(t_inf, t_sup, num=nb_points, endpoint=False)
        times[0] = t_min

        return times

