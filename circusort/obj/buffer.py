import numpy as np
import scipy


class Buffer(object):

    def __init__(self, sampling_rate, snippet_duration, data=None, alignment=True, factor=5):
        self.sampling_rate = sampling_rate
        self.alignment = alignment
        self.snippet_duration = snippet_duration

        self._spike_width_ = int(self.sampling_rate * self.snippet_duration * 1e-3)
        if np.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width = (self._spike_width_ - 1) // 2
        self._2_width = 2 * self._width
        self._limits = None

        if self.alignment:
            self.factor = factor
            self._cdata = np.linspace(-self._width, self._width, self.factor * self._spike_width_)
            self._xdata = np.arange(-self._2_width, self._2_width + 1)
            self._xoff = len(self._cdata) / 2.0

        self.data = data

    @property
    def nb_samples(self):
        if self.data is None:
            return 0
        else:
            return self.data.shape[0]

    @property
    def snippet_limits(self):
        if self._limits is None:
            if self.alignment:
                self._limits = (self._2_width, self.nb_samples - self._2_width)
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

    def update(self, data):
        self.data = data
        self._limits = None

    def median(self):
        return np.median(self.data, 1)

    def get_snippet(self, channels, peak, peak_type='negative', ref_channel=None):
        
        if len(channels) == 1:
            return self.get_waveform(channels[0], peak, peak_type)
        else:
            if self.alignment:

                k_min = peak - self._2_width
                k_max = peak + self._2_width + 1
                zdata = self.data[k_min:k_max, channels]
                ydata = np.arange(len(channels))

                f = scipy.interpolate.RectBivariateSpline(self._xdata, ydata, zdata, s=0, ky=min(len(ydata) - 1, 3))
                if peak_type == 'negative':
                    rmin = float(np.argmin(f(self._cdata, ref_channel)[:, 0]) - self._xoff) / 5.0
                elif peak_type == 'positive':
                    rmin = float(np.argmax(f(self._cdata, ref_channel)[:, 0]) - self._xoff) / 5.0
                ddata = np.linspace(rmin - self._width, rmin + self._width, self._spike_width_)
                data = f(ddata, ydata).astype(np.float32)
            else:
                data = self.data[peak - self._width:peak + self._width + 1, channels]

            return data

    def get_waveform(self, channel, peak, peak_type='negative'):
        if self.alignment:
            ydata = self.data[peak - self._2_width:peak + self._2_width + 1, channel]
            f = scipy.interpolate.UnivariateSpline(self._xdata, ydata, s=0)
            if peak_type == 'negative':
                rmin = float(np.argmin(f(self._cdata)) - self._xoff) / self.factor
            elif peak_type == 'positive':
                rmin = float(np.argmax(f(self._cdata)) - self._xoff) / self.factor
            ddata = np.linspace(rmin - self._width, rmin + self._width, self._spike_width_)

            data = f(ddata).astype(np.float32)
        else:
            data = self.data[peak - self._width:peak + self._width + 1, channel]
        
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

        if is_neg:
            is_neg = 'negative'
        else:
            is_neg = 'postive'

        return channels[channel], is_neg