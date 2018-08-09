import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.interpolate


class Snippet(object):

    def __init__(self, data, time_step=None, channel=None, channels=None,
                 sampling_rate=None, probe=None):

        self._data = data

        self._time_step = time_step
        self._channel = channel
        self._channels = channels

        self._sampling_rate = sampling_rate
        self._probe = probe

        self._is_aligned = False
        self._aligned_time_step = None
        self._aligned_data = None

        # TODO complete.

    @property
    def _nb_time_steps(self):

        # TODO check value in `__init__`.

        return (self._data.shape[0] - 1) // 2 + 1

    @property
    def _nb_extended_time_steps(self):

        # TODO check value in `__init__`.

        return self._data.shape[0]

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
    def _width(self):

        return (self._nb_time_steps - 1) // 2

    @property
    def _extended_width(self):

        return (self._nb_extended_time_steps - 1) // 2

    @property
    def _minimum_time_step(self):

        return self._time_step - self._width

    @property
    def _minimum_extended_time_step(self):

        return self._time_step - self._extended_width

    @property
    def _maximum_time_step(self):

        return self._time_step + self._width

    @property
    def _maximum_extended_time_step(self):

        return self._time_step + self._extended_width

    @property
    def _time_steps(self):

        return np.arange(self._minimum_time_step, self._maximum_time_step + 1)

    @property
    def _extended_time_steps(self):

        return np.arange(self._minimum_extended_time_step, self._maximum_extended_time_step + 1)

    def _densified_time_steps(self, factor=5):

        num = factor * self._nb_time_steps

        return np.linspace(self._minimum_time_step, self._maximum_time_step, num=num)

    def _aligned_time_steps(self, time_step):

        return (self._time_steps - self._time_step) + time_step

    def align(self, peak_type='negative', factor=5):

        if not self._is_aligned:
            if self._is_bivariate:
                self._aligned_time_step, self._aligned_data = self._align_bivariate(peak_type=peak_type, factor=factor)
            else:
                self._aligned_time_step, self._aligned_data = self._align_univariate(peak_type=peak_type, factor=factor)
            self._is_aligned = True

        return

    def _align_bivariate(self, peak_type='negative', factor=5):

        # Interpolate data.
        x = self._extended_time_steps
        y = self._channels
        z = self._data
        kx = 3  # i.e. x-degree of the bivariate spline  # TODO check if correct.
        ky = 1  # i.e. y-degree of the bivariate spline  # TODO check if correct.
        s = 0  # i.e. interpolate through all the data
        f = scipy.interpolate.RectBivariateSpline(x, y, z, kx=kx, ky=ky, s=s)

        # Find central time step.
        x = self._densified_time_steps(factor=factor)
        y = self._channel
        z = f(x, y)
        if peak_type == 'negative':
            index = np.argmin(z)
        elif peak_type == 'positive':
            index = np.argmax(z)
        else:
            raise NotImplementedError()
        central_time_step = x[index]

        # TODO remove the following line.
        print(central_time_step - self._time_step)

        # Align data.
        x = self._aligned_time_steps(central_time_step)
        y = self._channels
        aligned_data = f(x, y)
        aligned_data = aligned_data.astype('float32')

        return central_time_step, aligned_data

    def _align_univariate(self, peak_type='negative', factor=5):

        # Interpolate data.
        x = self._extended_time_steps
        y = self._data
        # s = 0  # i.e. interpolate through all the data
        s = 2
        f = scipy.interpolate.UnivariateSpline(x, y, s=s)

        # Find central time step.
        x = self._densified_time_steps(factor=factor)
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
        central_index = (len(aligned_data) - 1) // 2
        if np.argmin(aligned_data) != central_index:
            fig, ax = plt.subplots()
            ax.plot(self._extended_time_steps, self._data, color='C0')
            ax.plot(x, aligned_data, color='C1')
            fig.savefig("/tmp/waveforms/{}.pdf".format(int(1e+6 * np.random.uniform())))
            plt.close(fig)

        return central_time_step, aligned_data

    def to_array(self):

        if self._is_aligned:
            array = np.transpose(self._aligned_data)
        else:
            array = np.transpose(self._data)

        return array
