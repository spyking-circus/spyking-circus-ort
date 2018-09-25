import matplotlib.pyplot as plt
import numpy as np

from circusort.obj.snippet import Snippet
from circusort.io.probe import load_probe
from circusort.io.datafile import load_datafile, create_datafile
from circusort.utils import compute_snippet_width, compute_maximum_snippet_jitter


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

    @property
    def length(self):

        return float(self._nb_samples) / self._sampling_rate

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

    def get_snippet(self, channels, peak_time_step, ref_channel=None, peak_duration=5.0, peak_jitter=1.0):
        """Extract data snippet.

        Arguments:
            channels: iterable
                The channels to extract.
            peak_time_step: integer
                The time step of the peak.
            ref_channel: integer (optional)
                The channel of reference.
                The default value is None.
            peak_duration: float (optional)
                The duration of the peak [ms].
                The default value is 5.0.
            peak_jitter: float(optional)
                The maximum time jitter of the peak.
                The default value is 1.0.
        Return:
            snippet: circusort.obj.Snippet
                The extracted data snippet.
        """

        snippet_width = compute_snippet_width(peak_duration, self._sampling_rate)
        half_width = (snippet_width - 1) // 2
        max_jitter = compute_maximum_snippet_jitter(peak_jitter, self._sampling_rate)
        extended_half_width = half_width + max_jitter

        ts_min = peak_time_step - extended_half_width
        ts_max = peak_time_step + extended_half_width
        data = self._data.take(channels=channels, ts_min=ts_min, ts_max=ts_max)

        snippet = Snippet(data, width=half_width, jitter=max_jitter, time_step=peak_time_step, channel=ref_channel,
                          channels=channels, sampling_rate=self._sampling_rate, probe=self._probe)

        return snippet

    def plot(self, ax=None, channels=None, **kwargs):

        if ax is None:
            _, ax = plt.subplots(ncols=2)

        channel_colors = self._probe.get_channel_colors(selection=channels)

        self._data.plot(ax=ax[0], colors=channel_colors, **kwargs)
        self._probe.plot(ax=ax[1], colors=channel_colors)

        return
