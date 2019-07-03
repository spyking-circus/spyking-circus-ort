import numpy as np

from circusort.obj.snippet import Snippet
from circusort.utils import compute_snippet_width, compute_maximum_snippet_jitter


class Buffer(object):

    def __init__(self, sampling_rate, snippet_duration, snippet_jitter,
                 data=None, offset=0, alignment=True, factor=5, probe=None, hanning_filtering=False):

        self.sampling_rate = sampling_rate
        self.alignment = alignment
        self.snippet_duration = snippet_duration
        self.snippet_jitter = snippet_jitter

        self._spike_width_ = compute_snippet_width(self.snippet_duration, self.sampling_rate)
        self._width = (self._spike_width_ - 1) // 2
        self._jitter = compute_maximum_snippet_jitter(self.snippet_jitter, self.sampling_rate)
        self._extended_width = self._width + self._jitter
        self._limits = None
        self.hanning_filtering = hanning_filtering
        if self.hanning_filtering:
            self.filter = np.hanning(self._spike_width_)

        if self.alignment:
            self.factor = factor

        self._probe = probe

        self.data = data
        self._offset = offset

    @property
    def nb_samples(self):

        if self.data is None:
            nb_samples = 0
        else:
            nb_samples = self.data.shape[0]

        return nb_samples

    @property
    def snippet_limits(self):

        if self._limits is None:
            if self.alignment:
                self._limits = (self._extended_width, self.nb_samples - self._extended_width)
            else:
                self._limits = (self._width, self.nb_samples - self._width)

        return self._limits

    @property
    def temporal_width(self):

        return self._spike_width_
    
    def get_safety_time(self, safety_time='auto'):

        if safety_time == 'auto':
            safety_time = self._spike_width_ // 3
        else:
            safety_time = int(int(safety_time) * self.sampling_rate * 1e-3)

        return safety_time

    def valid_peaks(self, peaks):

        t_min, t_max = self.snippet_limits
        if np.iterable(peaks):
            return (peaks >= t_min) & (peaks < t_max)
        else:
            return (peaks >= t_min) and (peaks < t_max)

    def update(self, data, offset=0):

        self.data = data
        self._offset = offset

        self._limits = None

        return

    def median(self):

        return np.median(self.data, 1)

    def get_snippet(self, channels, peak, peak_type='negative', ref_channel=None, sigma=0):

        ts_min = peak - self._extended_width
        ts_max = peak + self._extended_width
        data = self.data[ts_min:ts_max + 1, channels]
        time_step = self._offset + peak
        snippet = Snippet(data, width=self._width, jitter=self._jitter, time_step=time_step, channel=ref_channel,
                          channels=channels, sampling_rate=self.sampling_rate, probe=self._probe)
        if self.alignment:
            snippet.align(peak_type=peak_type, factor=self.factor, sigma=sigma)

        if self.hanning_filtering:
            snippet.filter(self.filter)

        return snippet

    def get_waveform(self, channel, peak, peak_type='negative', sigma=0.0):

        ts_min = peak - self._extended_width
        ts_max = peak + self._extended_width
        data = self.data[ts_min:ts_max + 1, channel]
        time_step = self._offset + peak
        snippet = Snippet(data, width=self._width, jitter=self._jitter, time_step=time_step, channel=channel,
                          channels=np.array([channel]), sampling_rate=self.sampling_rate, probe=self._probe)
        if self.alignment:
            snippet.align(peak_type=peak_type, factor=self.factor, sigma=sigma)

        if self.hanning_filtering:
            snippet.filter(self.filter)

        data = snippet.to_array()

        return data

    def get_best_channel(self, channels, peak, peak_type='negative'):

        if peak_type == 'negative':
            channel = int(np.argmin(self.data[peak, channels]))
            is_neg = True
        elif peak_type == 'positive':
            channel = int(np.argmax(self.data[peak, channels]))
            is_neg = False
        elif peak_type == 'both':
            v_max = np.max(self.data[peak, channels])
            v_min = np.min(self.data[peak, channels])
            if np.abs(v_max) > np.abs(v_min):
                channel = int(np.argmax(self.data[peak, channels]))
                is_neg = False
            else:
                channel = int(np.argmin(self.data[peak, channels]))
                is_neg = True
        else:
            raise NotImplementedError()

        if is_neg:
            is_neg = 'negative'
        else:
            is_neg = 'positive'

        return channels[channel], is_neg
