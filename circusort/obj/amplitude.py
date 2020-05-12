import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
import numpy as np
import os

from circusort.utils.path import normalize_path


class Amplitude(object):
    """The amplitude of a cell through time."""
    # TODO complete docstring.

    def __init__(self, amplitudes, times, t_min=None, t_max=None):
        """Initialization."""
        # TODO complete docstring.

        mask = np.ones_like(times, dtype=np.bool)
        if t_min is not None:
            mask = np.logical_and(mask, t_min <= times)
        if t_max is not None:
            mask = np.logical_and(mask, times <= t_max)

        if len(mask) > 0:
            self.amplitudes = amplitudes[mask]
            self.times = times[mask]
        else:
            self.amplitudes = np.array(amplitudes)
            self.times = np.array(times)

        assert len(self.times) == len(self.amplitudes), "Times and Amplitudes should have the same length"

        if len(self.times) > 0:
            self.t_min = max(np.min(times), 0) if t_min is None else t_min
            self.t_max = np.max(times) if t_max is None else t_max
        else:
            self.t_min = 0 if t_min is None else t_min
            self.t_max = 0 if t_max is None else t_max

    def __iter__(self):

        return self.amplitudes.__iter__()

    def __len__(self):

        return len(self.times)

    def __str__(self):

        string = "<circusort.obj.amplitude.Amplitude t_min:{:.3f} t_max:{:.3f} len:{} a_min:{:.3f} a_max:{:.3f}>"
        string = string.format(self.t_min, self.t_max, len(self), np.min(self.amplitudes), np.max(self.amplitudes))

        return string

    @property
    def two_components(self):
        res = self.amplitudes.shape[0] == 2
        return res

    def append(self, amplitudes):
        assert isinstance(amplitudes, Amplitude), "Can only append Amplitude to Amplitude object"
        self.amplitudes = np.concatenate((self.amplitudes, amplitudes.amplitudes))
        self.times = np.concatenate((self.times, amplitudes.times))
        self.t_min = min(self.t_min, amplitudes.t_min)
        self.t_max = max(self.t_max, amplitudes.t_max)

    def slice(self, t_min=None, t_max=None):
        # TODO add docstring.

        if t_min is None:
            t_min = self.t_min
        if t_max is None:
            t_max = self.t_max

        idx = np.where((self.times >= t_min) & (self.times <= t_max))[0]

        amplitude = Amplitude(self.amplitudes[idx], self.times[idx], t_min=t_min, t_max=t_max)

        return amplitude

    def save(self, path):
        """Save amplitude to file.

        Parameters:
            path: string
                The path to the file in which to save the amplitude.
        """

        with h5py.File(path, mode='w') as file_:
            kwargs = {
                'shape': self.amplitudes.shape,
                'dtype': self.amplitudes.dtype,
                'data': self.amplitudes,
            }
            file_.create_dataset('amplitudes', **kwargs)
            kwargs = {
                'shape': self.times.shape,
                'dtype': self.times.dtype,
                'data': self.times,
            }
            file_.create_dataset('times', **kwargs)

        return

    def _plot(self, ax, set_ax=False, t_min=None, t_max=None, **kwargs):

        _ = kwargs  # Discard additional keyword arguments.

        x = self.times
        y = self.amplitudes

        if set_ax:
            x_min = np.min(x) if t_min is None else t_min
            x_max = np.max(x) if t_max is None else t_max
            ax.set_xlim(x_min, x_max)
            ax.set_xlabel(u"time (s)")
            ax.set_ylabel(u"amplitude (arb. unit)")
            ax.set_title(u"Amplitude")
        ax.plot(x, y)

        return

    def plot(self, output=None, ax=None, set_ax=False, **kwargs):
        # TODO add docstring.

        if output is not None and ax is None:
            plt.ioff()

        if ax is None:
            fig = plt.figure()
            gs = gds.GridSpec(1, 1)
            ax_ = fig.add_subplot(gs[0])
            self._plot(ax_, set_ax=True, **kwargs)
            gs.tight_layout(fig)
            if output is None:
                fig.show()
            else:
                path = normalize_path(output)
                if path[-4:] != ".pdf":
                    path = os.path.join(path, "amplitude.pdf")
                directory = os.path.dirname(path)
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                fig.savefig(path)
        else:
            self._plot(ax, set_ax=set_ax, **kwargs)

        return
