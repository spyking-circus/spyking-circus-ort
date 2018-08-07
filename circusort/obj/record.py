import numpy as np

from circusort.io.probe import load_probe
from circusort.io.datafile import load_datafile, create_datafile


class Record(object):
    """Data record."""

    def __init__(self, data_path, probe_path, sampling_rate=20e+3, dtype='int16', gain=0.1042):

        self._data_path = data_path
        self._probe_path = probe_path
        self._sampling_rate = sampling_rate
        self._dtype = dtype
        self._gain = gain

        self._probe = load_probe(self._probe_path)

        self._data = load_datafile(self._data_path, self._sampling_rate, self._nb_channels, self._dtype, gain=gain)

    @property
    def _nb_samples(self):

        return self._data.nb_samples

    @property
    def _nb_channels(self):

        return self._probe.total_nb_channels

    def copy(self, data_path, probe_path, channels=None, t_min=None, t_max=None, nb_time_steps_per_chunk=1024):
        """Copy data record.

        Arguments:
            data_path: string
            probe_path: string
            channels: none | iterable (optional)
                These values define the channels to copy.
                The default value is None.
            t_min: none | iterable (optional)
                This value defines where to start the copy in time [s].
                The default value is None.
            t_max: none | iterable (optional)
                This value defines where to end the copy in time [s].
                The default value is None.
            nb_time_steps_per_chunk: integer
                The number of time steps per chunk.
                The default value is 1024.
        """

        # Compute minimum time step.
        if t_min is None:
            ts_min = 0
        else:
            ts_min = int(np.ceil(t_min * self._sampling_rate))
            assert 0 <= ts_min, "ts_min: {}".format(ts_min)
        # Compute maximum time step.
        if t_max is None:
            ts_max = self._nb_samples - 1
        else:
            ts_max = int(np.floor(t_max * self._sampling_rate))
            assert ts_max <= self._nb_samples - 1, "ts_max: {}".format(ts_max)

        # Initialize copied data.
        nb_time_steps = ts_max - ts_min + 1
        nb_channels = len(channels) if channels is not None else self._nb_channels
        sampling_rate = self._sampling_rate
        dtype = self._dtype
        copied_data = create_datafile(data_path, nb_time_steps, nb_channels, sampling_rate, dtype)

        # Compute number of chunks.
        if nb_time_steps % nb_time_steps_per_chunk == 0:
            nb_chunks = int(np.floor(float(nb_time_steps) / float(nb_time_steps_per_chunk)))
            nb_time_steps_in_last_chunk = 0
        else:
            nb_chunks = int(np.floor(float(nb_time_steps) / float(nb_time_steps_per_chunk))) + 1
            nb_time_steps_in_last_chunk = nb_time_steps % nb_time_steps_per_chunk

        # Copy data.
        for k in range(0, nb_chunks - 1):
            ts_start = ts_min + (k + 0) * nb_time_steps_per_chunk
            ts_end = ts_min + (k + 1) * nb_time_steps_per_chunk - 1
            data = self._data.take(channels=channels, ts_min=ts_start, ts_max=ts_end)
            ts_start = (k + 0) * nb_time_steps_per_chunk
            ts_end = (k + 1) * nb_time_steps_per_chunk - 1
            copied_data.put(data, ts_min=ts_start, ts_max=ts_end)
        for k in range(nb_chunks - 1, nb_chunks):
            ts_start = ts_min + k * nb_time_steps_per_chunk
            ts_end = ts_min + k * nb_time_steps_per_chunk + nb_time_steps_in_last_chunk - 1
            data = self._data.take(channels=channels, ts_min=ts_start, ts_max=ts_end)
            ts_start = k * nb_time_steps_per_chunk
            ts_end = k * nb_time_steps_per_chunk + nb_time_steps_in_last_chunk - 1
            copied_data.put(data, ts_min=ts_start, ts_max=ts_end)

        # Copy probe.
        copied_probe = self._probe.copy()
        # Keep channels of interest only.
        copied_probe.keep(channels)
        # Save copied probe.
        copied_probe.save(probe_path)

        return
