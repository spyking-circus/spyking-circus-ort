import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.interpolate


class Snippet(object):

    def __init__(self, data, width=None, jitter=None, time_step=None, channel=None, channels=None,
                 sampling_rate=None, probe=None):

        self._data = data

        nb_time_steps = data.shape[0]

        if width is not None:
            self._width = width
            if jitter is not None:
                self._jitter = jitter
                assert nb_time_steps
            else:
                self._jitter = (nb_time_steps - 1 - 2 * self._width) // 2
        else:
            if jitter is not None:
                self._jitter = jitter
            else:
                self._jitter = (nb_time_steps - 1) // 4
            self._width = (nb_time_steps - 1 - 2 * self._jitter) // 2

        self._time_step = time_step
        self._channel = channel
        self._channels = channels

        self._sampling_rate = sampling_rate
        self._probe = probe

        self._is_aligned = False
        self._aligned_time_step = None
        self._aligned_data = None

    @property
    def _nb_time_steps(self):

        # TODO check value in `__init__`.

        return self._width + 1 + self._width

    @property
    def _nb_extended_time_steps(self):

        # TODO check value in `__init__`.

        return self._jitter + self._width + 1 + self._width + self._jitter

    @property
    def _nb_jittered_time_steps(self):

        return self._jitter + 1 + self._jitter

    @property
    def _is_bivariate(self):

        return self._data.ndim == 2

    @property
    def _nb_channels(self):

        # TODO check consistency with `channels` in `__init__`.

        if self._is_bivariate:
            nb_channels = self._data.shape[1]
        else:
            nb_channels = 1

        return nb_channels

    @property
    def _extended_width(self):

        return self._width + self._jitter

    @property
    def _minimum_time_step(self):

        return self._time_step - self._width

    @property
    def _minimum_extended_time_step(self):

        return self._time_step - self._extended_width

    @property
    def _minimum_jittered_time_step(self):

        return self._time_step - self._jitter

    @property
    def _maximum_time_step(self):

        return self._time_step + self._width

    @property
    def _maximum_extended_time_step(self):

        return self._time_step + self._extended_width

    @property
    def _maximum_jittered_time_step(self):

        return self._time_step + self._jitter

    @property
    def _time_steps(self):

        return np.arange(self._minimum_time_step, self._maximum_time_step + 1)

    @property
    def _extended_time_steps(self):

        return np.arange(self._minimum_extended_time_step, self._maximum_extended_time_step + 1)

    def _jittered_time_steps(self, factor=5):

        num = factor * self._nb_jittered_time_steps

        return np.linspace(self._minimum_jittered_time_step, self._maximum_jittered_time_step, num=num)

    def _aligned_time_steps(self, time_step):

        return (self._time_steps - self._time_step) + time_step

    @property
    def _amplitude(self):

        amplitude = 2.0 * np.amax(np.abs(self._data))

        return amplitude

    def align(self, peak_type='negative', factor=5, sigma=0, degree=3):

        if not self._is_aligned:
            if self._is_bivariate:
                self._aligned_time_step, self._aligned_data = self._align_bivariate(peak_type=peak_type, factor=factor,
                                                                                    sigma=sigma, degree=degree)
            else:
                self._aligned_time_step, self._aligned_data = self._align_univariate(peak_type=peak_type, factor=factor,
                                                                                     sigma=sigma, degree=degree)
            self._is_aligned = True

        return

    def _approximate_bivariate(self, sigma=0.0, degree=3):
        """Bivariate approximation of the snippet.

        Arguments:
            sigma: float (optional)
                Smoothing factor. Ideally, it should be an estimate of the standard deviation for all the data points.
                If 0.0, spline will interpolate through all the data points (i.e. no smoothing at all).
                The default value is 0.0.
            degree: integer (optional)
                Degree of the spline along the x-axis.
                The default value is 3.
        Return:
            f: scipy.interpolate.RectBivariateSpline
                Bivariate spline approximation.
        """

        nb_data_points = self._data.size

        x = self._extended_time_steps
        y = self._channels
        z = self._data
        kx = degree  # i.e. degree of the bivariate spline along the x-axis
        ky = 1  # i.e. degree of the bivariate spline along the y-axis
        if sigma > 0:
            s = nb_data_points * sigma  # i.e. smoothing factor
        else:
            s = 0
        f = scipy.interpolate.RectBivariateSpline(x, y, z, kx=kx, ky=ky, s=s)

        return f

    def _align_bivariate(self, peak_type='negative', factor=5, sigma=0.0, degree=3):

        # Interpolate data.
        f = self._approximate_bivariate(sigma=sigma, degree=degree)

        # Find central time step.
        x = self._jittered_time_steps(factor=factor)
        y = self._channel
        z = f(x, y)
        if peak_type == 'negative':
            index = np.argmin(z)
        elif peak_type == 'positive':
            index = np.argmax(z)
        else:
            raise NotImplementedError()
        central_time_step = x[index]

        # Align data.
        x = self._aligned_time_steps(central_time_step)
        y = self._channels
        aligned_data = f(x, y)
        aligned_data = aligned_data.astype('float32')

        return central_time_step, aligned_data

    def _approximate_univariate(self, sigma=0.0, degree=3):
        """Univariate approximation of the snippet.

        Arguments:
            sigma: float (optional)
                Smoothing factor. Ideally, it should be an estimate of the standard deviation for all the data points.
                If 0.0, spline will interpolate through all the data points (i.e. no smoothing at all).
                The default value is 0.0.
            degree: integer (optional)
                Degree of the spline.
                The default value is 3.
        Return:
            f: scipy.interpolate.RectBivariateSpline
                Bivariate spline approximation.
        """

        nb_data_points = self._data.size

        x = self._extended_time_steps
        y = self._data
        k = degree  # i.e. degree of the univariate spline
        if sigma > 0:
            s = float(nb_data_points) * sigma
        else:
            s = 0
        f = scipy.interpolate.UnivariateSpline(x, y, k=k, s=0)

        return f

    def _align_univariate(self, peak_type='negative', factor=5, sigma=0.0, degree=3):

        # Interpolate data.
        f = self._approximate_univariate(sigma=sigma, degree=degree)

        # Find central time step.
        x = self._jittered_time_steps(factor=factor)
        y = f(x)
        if peak_type == 'negative':
            index = np.argmin(y)
        elif peak_type == 'positive':
            index = np.argmax(y)
        else:
            raise NotImplementedError()
        central_time_step = x[index]

        # Align data.
        x = self._aligned_time_steps(central_time_step)
        aligned_data = f(x)
        aligned_data = aligned_data.astype('float32')

        # TODO remove the following line.
        import os
        if not os.path.isdir("/tmp/waveforms"):
            os.makedirs("/tmp/waveforms")
        central_index = (len(aligned_data) - 1) // 2
        if np.argmin(aligned_data) != central_index:
            fig, ax = plt.subplots()
            ax.plot(self._extended_time_steps, self._data, color='C0')
            ax.plot(x, aligned_data, color='C1')
            ax.set_title("c{} ts{}".format(self._channel, self._time_step))
            fig.savefig("/tmp/waveforms/c{}_ts{}.pdf".format(self._channel, self._time_step))
            plt.close(fig)

        return central_time_step, aligned_data

    def to_array(self):

        if self._is_aligned:
            array = np.transpose(self._aligned_data)
        else:
            array = np.transpose(self._data)

        return array

    def filter(self, my_filter):
        self._data = (self._data.T * my_filter).T

    def plot(self, ax=None, **kwargs):

        if ax is None:
            _, ax = plt.subplots()

        data = 0.9 * self._data / self._amplitude if self._amplitude > 0.0 else self._data

        ax.axvline(x=self._time_step, color='grey', linestyle='dashed')

        x = self._extended_time_steps
        if self._is_bivariate:
            for k in range(0, self._nb_channels):
                y_offset = float(k)
                y = data[:, k] + y_offset
                ax.plot(x, y, **kwargs)
        else:
            y = data
            ax.plot(x, y, **kwargs)

        return ax

    def plot_aligned(self, ax=None, **kwargs):

        if ax is None:
            _, ax = plt.subplots()

        data = 0.9 * self._aligned_data / self._amplitude if self._amplitude > 0.0 else self._aligned_data

        ax.axvline(x=self._aligned_time_step, color='grey', linestyle='dashed')

        x = self._aligned_time_steps(self._aligned_time_step)
        if self._is_bivariate:
            for k in range(0, self._nb_channels):
                y_offset = float(k)
                y = data[:, k] + y_offset
                ax.plot(x, y, **kwargs)
        else:
            y = data
            ax.plot(x, y, **kwargs)

        return ax

    def plot_jittered(self, ax=None, sigma=0.0, degree=3, **kwargs):

        if ax is None:
            _, ax = plt.subplots()

        x = self._jittered_time_steps()
        if self._is_bivariate:
            f = self._approximate_bivariate(sigma=sigma, degree=degree)
            y = self._channels
            data = 0.9 * f(x, y) / self._amplitude if self._amplitude > 0.0 else f(x, y)
            for k in range(0, self._nb_channels):
                y_offset = float(k)
                y = data[:, k] + y_offset
                ax.plot(x, y, **kwargs)
        else:
            f = self._approximate_univariate(sigma=sigma, degree=degree)
            data = 0.9 * f(x) / self._amplitude if self._amplitude > 0.0 else f(x)
            y = data
            ax.plot(x, y, **kwargs)

        return ax
