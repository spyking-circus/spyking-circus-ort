import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from circusort.utils.path import normalize_path


class Train(object):
    # TODO add docstring

    def __init__(self, times):
        # TODO add docstring.

        self.times = times

    def save(self, path):
        """Save train to file.

        Parameters:
            path: string
                The path to the file in which to save the train.
        """

        file_ = h5py.File(path, mode='w')
        file_.create_dataset('times', shape=self.times.shape, dtype=self.times.dtype, data=self.times)
        file_.close()

        return

    @property
    def nb_times(self):
        # TODO add docstring.

        nb_times = self.times.size

        return nb_times

    def plot(self, output=None, t_min=0.0, t_max=10.0, **kwargs):
        # TODO add docstring.

        _ = kwargs  # Discard additional keyword arguments.

        is_selected = np.logical_and(t_min <= self.times, self.times <= t_max)
        x = self.times[is_selected]
        y = np.zeros_like(x)
        x_min = t_min
        x_max = t_max

        if output is not None:
            plt.ioff()

        fig, ax = plt.subplots()
        ax.set_xlim(x_min, x_max)
        ax.scatter(x, y)  # TODO control the radius of the somas of the cells.
        ax.set_yticks([])
        ax.set_xlabel(u"time (s)")
        ax.set_title(u"Train")
        fig.tight_layout()

        if output is None:
            plt.show()
        else:
            path = normalize_path(output)
            if path[-4:] != ".pdf":
                path = os.path.join(path, "train.pdf")
            directory = os.path.dirname(path)
            if not os.path.isdir(directory):
                os.makedirs(directory)
            fig.savefig(path)

        return

    # TODO complete.
