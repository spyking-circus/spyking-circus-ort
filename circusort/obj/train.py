import h5py
import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
import numpy as np
import os

from circusort.utils.path import normalize_path


class Train(object):
    # TODO add docstring

    def __init__(self, times, t_min=None, t_max=None):
        # TODO add docstring.

        self.times = times
        self.times = self.times if t_min is None else self.times[t_min <= self.times]
        self.times = self.times if t_max is None else self.times[self.times <= t_max]
        self.t_min = min(0.0, np.min(times)) if t_min is None else t_min
        self.t_max = np.max(times) if t_max is None else t_max

    def __len__(self):

        return len(self.times)

    def __iter__(self):
        
        return self.times.__iter__()

    @property
    def nb_times(self):

        return self.times.size

    @property
    def mean_rate(self):

        return len(self) / (self.t_max - self.t_min)

    def reverse(self):
        # TODO add docstring.

        # TODO improve method with two additional attributes: start_time and end_time.
        times = self.t_min + ((self.t_max - self.t_min) - (self.times - self.t_min))
        train = Train(times, t_min=self.t_min, t_max=self.t_max)

        return train

    def slice(self, t_min=None, t_max=None):
        # TODO add docstring.

        times = self.times
        if t_min is None:
            t_min = self.t_min
        elif isinstance(t_min, float):
            times = times[t_min <= times]
        if t_max is None:
            t_max = self.t_max
        elif isinstance(t_max, float):
            times = times[times <= t_max]

        train = Train(times, t_min=t_min, t_max=t_max)

        return train

    def save(self, path):
        """Save train to file.

        Parameters:
            path: string
                The path to the file in which to save the train.
        """
        with h5py.File(path, mode='w') as file_:
            file_.create_dataset('times', shape=self.times.shape, dtype=self.times.dtype, data=self.times)

        return

    def rate(self, time_bin=1):

        bins = np.arange(self.t_min, self.t_max, time_bin)
        x, y = np.histogram(self.times, bins=bins)

        return x/time_bin

    def _plot(self, ax, t_min=0.0, t_max=10.0, offset=0, **kwargs):

        _ = kwargs  # Discard additional keyword arguments.

        t_min = self.t_min + t_min
        t_max = self.t_min + t_max

        is_selected = np.logical_and(t_min <= self.times, self.times <= t_max)
        x = self.times[is_selected]
        y = offset * np.ones_like(x)
        x_min = t_min
        x_max = t_max

        ax.set_xlim(x_min, x_max)
        ax.scatter(x, y)  # TODO control the radius of the somas of the cells.
        ax.set_yticks([])
        ax.set_xlabel(u"time (s)")
        ax.set_title(u"Train")

        return

    def plot(self, output=None, ax=None, **kwargs):
        # TODO add docstring.

        if output is not None and ax is None:
            plt.ioff()

        if ax is None:
            fig = plt.figure()
            gs = gds.GridSpec(1, 1)
            ax_ = fig.add_subplot(gs[0])
            self._plot(ax_, **kwargs)
            gs.tight_layout(fig)
            if output is None:
                fig.show()
            else:
                path = normalize_path(output)
                if path[-4:] != ".pdf":
                    path = os.path.join(path, "train.pdf")
                directory = os.path.dirname(path)
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                fig.savefig(path)
        else:
            self._plot(ax, **kwargs)

        return

    def compute_fp_rates(self, train, jitter=2e-3, t_min=None, t_max=None):
        """Compute the false positive rates.

        Return the false positive rates between a given spike train and
        another spike train. All rates are established up to a certain jitter,
        expressed in time steps.

        The function returns a tuple with two elements, the two false positive
        rates (1st train compared to the 2nd, and 2nd compared to the 1st one).

        Arguments:
            train: circusort.obj.Train
                The train with which the difference has to be computed.
            jitter: float (optional)
                The jitter to use to compare the trains.
                The default value is 2e-3.
            t_min: none | float (optional)
                The start time of the window to use for the computation.
                The default value is None.
            t_max: none | float (optional)
                The end time of the window to use for the computation.
                The default value is None.
        Return:
            fp_rates: numpy.ndarray
                The computed false positive rates.
        """

        if t_min is None:
            t_min = max(self.t_min, train.t_min)
        if t_max is None:
            t_max = min(self.t_max, train.t_max)

        message = "Impossible to compare trains with disjoint temporal support."
        assert t_min <= t_max, message

        train_1 = self.slice(t_min=t_min, t_max=t_max)
        train_2 = train.slice(t_min=t_min, t_max=t_max)

        # Compute the true positive rate of the 1st train compared to the 2nd.
        count = 0
        for spike in train_1:
            idx = np.where(np.abs(train_2.times - spike) < jitter)[0]
            if len(idx) > 0:
                count += 1
        if len(train_1) > 0:
            tp_rate_1 = float(count) / float(len(train_1))
        else:
            tp_rate_1 = 0.0

        # Compute the true positive rate of the 2nd train compared to the 1st.
        count = 0
        for spike in train_2:
            idx = np.where(np.abs(train_1.times - spike) < jitter)[0]
            if len(idx) > 0:
                count += 1
        if len(train_2) > 0:
            tp_rate_2 = float(count) / float(len(train_2))
        else:
            tp_rate_2 = 0.0

        fp_rate_1 = 1.0 - tp_rate_1
        fp_rate_2 = 1.0 - tp_rate_2

        fp_rates = np.array([fp_rate_1, fp_rate_2])

        return fp_rates

    def compute_difference(self, train, **kwargs):
        """Compute the difference between two trains.

        Argument:
            train: circusort.obj.Train
                The train with which the difference has to be computed.
        Return:
            difference: float
                The difference between the two trains (as a value between 0 and 1).

        See also:
            circusort.obj.Train.compute_fp_rates for additional keyword
            arguments.
        """

        fp_rates = self.compute_fp_rates(train, **kwargs)
        difference = np.mean(fp_rates)

        return difference

    # TODO complete.
