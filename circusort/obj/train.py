import h5py
import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
import numpy as np
import os

from circusort.utils.path import normalize_path


class Train(object):

    def __init__(self, times, t_min=None, t_max=None):

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

        # TODO improve method with two additional attributes: start_time and end_time.
        times = self.t_min + ((self.t_max - self.t_min) - (self.times - self.t_min))
        train = Train(times, t_min=self.t_min, t_max=self.t_max)

        return train

    def slice(self, t_min=None, t_max=None):

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

    def sample(self, size=None):

        if size is None:
            size = 1
        if len(self) < size:
            size = len(self)

        indices = np.random.choice(len(self), size=size, replace=False)
        indices = np.sort(indices)
        times = self.times[indices]

        return times

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

    def check_temporal_support(self, train, t_min=None, t_max=None):
        """Check temporal support."""

        if t_min is None:
            t_min = max(self.t_min, train.t_min)
        if t_max is None:
            t_max = min(self.t_max, train.t_max)

        message = "Impossible to compare trains with disjoint temporal support."
        assert t_min <= t_max, message

        train_pred = self.slice(t_min=t_min, t_max=t_max)
        train_true = train.slice(t_min=t_min, t_max=t_max)

        return train_pred, train_true

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

        train_1, train_2 = self.check_temporal_support(train, t_min=t_min, t_max=t_max)

        # Compute the true positive rate of the 1st train compared to the 2nd.
        count = 0
        for spike in train_1:
            idx = np.where(np.abs(train_2.times - spike) < jitter)[0]
            if len(idx) > 0:
                count += 1
        if len(train_1) > 0:
            tp_rate_1 = float(count) / float(len(train_1))
        else:
            tp_rate_1 = 1.0

        # Compute the true positive rate of the 2nd train compared to the 1st one.
        count = 0
        for spike in train_2:
            idx = np.where(np.abs(train_1.times - spike) < jitter)[0]
            if len(idx) > 0:
                count += 1
        if len(train_2) > 0:
            tp_rate_2 = float(count) / float(len(train_2))
        else:
            tp_rate_2 = 1.0

        fp_rate_1 = 1.0 - tp_rate_1
        fp_rate_2 = 1.0 - tp_rate_2

        fp_rates = np.array([fp_rate_1, fp_rate_2])

        return fp_rates

    def compute_true_positive(self, train, jitter=2e-3, t_min=None, t_max=None):
        """Compute the number of true positives."""

        train_pred, train_true = self.check_temporal_support(train, t_min=t_min, t_max=t_max)

        # Compute the number of true positives of the predicted train compared to the true one.
        nb_tp_pred = 0  # i.e. number of true positives
        for spike_time in train_pred:
            indices = np.where(np.abs(train_true.times - spike_time) < jitter)[0]
            if len(indices) > 0:  # i.e. this is a true positive
                nb_tp_pred += 1
        # Compute the number of true positives of the true train compared to the predicted one.
        nb_tp_true = 0  # i.e. number of true positives
        for spike_time in train_true:
            indices = np.where(np.abs(train_pred.times - spike_time) < jitter)[0]
            if len(indices) > 0:  # i.e. this is a true positive
                nb_tp_true += 1
        # Compute the number of true positives.
        nb_tp = np.mean([nb_tp_pred, nb_tp_true])

        return nb_tp

    def compute_false_positive(self, train, jitter=2e-3, t_min=None, t_max=None):
        """Compute the number of false positives."""

        train_pred, train_true = self.check_temporal_support(train, t_min=t_min, t_max=t_max)

        # Compute the number of false positives of the predicted train compared to the true one.
        nb_fp = 0  # i.e. number of false positives
        for spike_time in train_pred:
            indices = np.where(np.abs(train_true.times - spike_time) < jitter)[0]
            if len(indices) == 0:  # i.e. this is a false positive
                nb_fp += 1

        return nb_fp

    def collect_false_positives(self, train, jitter=2e-3, t_min=None, t_max=None):
        """Collect the false positives."""

        train_pred, train_true = self.check_temporal_support(train, t_min=t_min, t_max=t_max)

        # Collect the false positives of the predicted train compared to the true one.
        times = []  # times associated to false positives
        for time in train_pred:
            indices = np.where(np.abs(train_true.times - time) < jitter)[0]
            if len(indices) == 0:  # i.e. this is a false positive
                times.append(time)
        times = np.array(times)
        t_min = train_pred.t_min
        t_max = train_pred.t_max
        train = Train(times, t_min=t_min, t_max=t_max)

        return train

    def compute_false_negative(self, train, jitter=2e-3, t_min=None, t_max=None):
        """Compute the number of false negatives."""

        train_pred, train_true = self.check_temporal_support(train, t_min=t_min, t_max=t_max)

        # Compute the number of false negatives of the predicted train compared to the true one.
        nb_fn = 0  # i.e. number of false negatives
        for spike_time in train_true:
            indices = np.where(np.abs(train_pred.times - spike_time) < jitter)[0]
            if len(indices) == 0:  # i.e. this is a false negative
                nb_fn += 1

        return nb_fn

    def collect_false_negatives(self, train, jitter=2e-3, t_min=None, t_max=None):
        """Collect the false negatives."""

        train_pred, train_true = self.check_temporal_support(train, t_min=t_min, t_max=t_max)

        # Collect the false negatives of the predicted train compared to the true one.
        times = []  # i.e. times associated to false negatives
        for time in train_true:
            indices = np.where(np.abs(train_pred.times - time) < jitter)[0]
            if len(indices) == 0:  # i.e. this is a false negative
                times.append(time)
        times = np.array(times)
        t_min = train_true.t_min
        t_max = train_true.t_max
        train = Train(times, t_min=t_min, t_max=t_max)

        return train

    def compute_true_positive_rate(self, train, jitter=2e-3, t_min=None, t_max=None):
        """Compute the true positive rate."""

        train_pred, train_true = self.check_temporal_support(train, t_min=t_min, t_max=t_max)

        # Compute the true positive rate of the predicted train compared to the true one.
        nb_tp = train_pred.compute_true_positive(train_true, jitter=jitter, t_min=t_min, t_max=t_max)
        nb_p = len(train_true)  # i.e. number of positives
        if nb_p > 0:
            r_tp = float(nb_tp) / float(nb_p)
            r_tp = min(1.0, r_tp)  # correct (if necessary)
        else:
            r_tp = 1.0

        return r_tp

    def compute_false_negative_rate(self, train, jitter=2e-3, t_min=None, t_max=None):
        """Compute the false negative rate."""

        train_pred, train_true = self.check_temporal_support(train, t_min=t_min, t_max=t_max)

        # Compute the false negative rate of the predicted train compared to the true one.
        nb_fn = train_pred.compute_false_negative(train_true, jitter=jitter, t_min=t_min, t_max=t_max)
        nb_p = len(train_true)  # i.e. number of positives
        if nb_p > 0:
            r_fn = float(nb_fn) / float(nb_p)
            r_fn = min(1.0, r_fn)  # correct (if necessary)
        else:
            r_fn = 0.0

        return r_fn

    def compute_false_negative_count(self, train, jitter=2e-3, t_min=None, t_max=None, nb_bins=50):
        """Compute the false negative count."""

        train_pred, train_true = self.check_temporal_support(train, t_min=t_min, t_max=t_max)

        fn_train = train_pred.collect_false_negatives(train_true, jitter=jitter, t_min=t_min, t_max=t_max)
        fn_times = fn_train.times
        range_ = (
            t_min if t_min is not None else fn_times.min(),
            t_max if t_max is not None else fn_times.max(),
        )
        bin_values, bin_edges = np.histogram(fn_times, bins=nb_bins, range=range_)

        return bin_values, bin_edges

    def compute_positive_predictive_value(self, train, jitter=2e-3, t_min=None, t_max=None):
        """Compute the positive predictive value."""

        train_pred, train_true = self.check_temporal_support(train, t_min=t_min, t_max=t_max)

        # Compute the positive predictive value of the predicted train compared to the true one.
        nb_tp = train_pred.compute_true_positive(train_true, jitter=jitter, t_min=t_min, t_max=t_max)
        nb_t = len(train_pred)  # i.e. number of trues
        if nb_t > 0:
            v_pp = float(nb_tp) / float(nb_t)
            v_pp = min(1.0, v_pp)  # correct (if necessary)
        else:
            v_pp = 1.0

        return v_pp

    def compute_false_discovery_rate(self, train, jitter=2e-3, t_min=None, t_max=None):
        """Compute the false discovery rate."""

        train_pred, train_true = self.check_temporal_support(train, t_min=t_min, t_max=t_max)

        # Compute the false discovery rate of the predicted train compared to the true one.
        nb_fp = train_pred.compute_false_positive(train_true, jitter=jitter, t_min=t_min, t_max=t_max)
        nb_t = len(train_pred)  # i.e. number of trues
        if nb_t > 0:
            r_fd = float(nb_fp) / float(nb_t)
            r_fd = min(1.0, r_fd)  # correct (if necessary)
        else:
            r_fd = 0.0

        return r_fd

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

        # fp_rates = self.compute_fp_rates(train, **kwargs)
        # difference = np.mean(fp_rates)

        r_fn = self.compute_false_negative_rate(train, **kwargs)
        r_fd = self.compute_false_discovery_rate(train, **kwargs)
        difference = np.mean([r_fn, r_fd])

        return difference

    def cross_correlogram(self, other, bin_width=1.0, width=201.0):
        """Compute the cross-correlogram.

        Arguments:
            other: circusort.obj.Train
                The train with which to compute the cross-correlations.
            bin_width: float (optional)
                The bin width of the cross-correlogram (ms).
                The default value is 1.0.
            width: float (optional)
                The width of the cross-correlogram (ms).
                The default value is 201.0.
        Return:
            cross_correlogram: numpy.ndarray
        """

        nb_bins = int(np.ceil(width / bin_width))

        bin_counts = np.zeros(nb_bins)

        half_width = 0.5 * width * 1e-3

        self_times = np.sort(self.times)
        other_times = np.sort(other.times)

        nb_self_times = self_times.size
        nb_other_times = other_times.size

        if nb_self_times > 0 and nb_other_times > 0:
            # 1.
            self_index = 0
            # Find the index of the earliest time which lie in the specified range of lags.
            start_other_index = 0
            while start_other_index < nb_other_times and \
                    other_times[start_other_index] < self_times[self_index] - half_width:
                start_other_index += 1
            # Find the index of the latest time which lie in the specified range of lags.
            end_other_index = start_other_index
            while end_other_index < nb_other_times and \
                    other_times[end_other_index] < self_times[self_index] + half_width:
                end_other_index += 1
            # For each index of a time which lie in the specified range of lags...
            for other_index in range(start_other_index, end_other_index):
                diff = other_times[other_index] - self_times[self_index]
                bin_index = int(np.floor((diff + half_width) * 1e+3 / bin_width))
                bin_counts[bin_index] += 1
            # 2.
            for self_index in range(1, nb_self_times):
                # Update the index of the earliest time which lie in the specified range of lags.
                while start_other_index < nb_other_times and \
                        other_times[start_other_index] < self_times[self_index] - half_width:
                    start_other_index += 1
                # Update the index of the latest time which lie in the specified range of lags.
                while end_other_index < nb_other_times and \
                        other_times[end_other_index] < self_times[self_index] + half_width:
                    end_other_index += 1
                # For each index of a time which lie in the specified range of lags...
                for other_index in range(start_other_index, end_other_index):
                    diff = other_times[other_index] - self_times[self_index]
                    bin_index = int(np.floor((diff + half_width) * 1e+3 / bin_width))
                    bin_counts[bin_index] += 1

        # TODO normalize the cross-correlogram (i.e. normalize values instead of bin counts)?

        bin_edges = np.linspace(-half_width * 1e+3, +half_width * 1e+3, num=nb_bins+1)

        return bin_counts, bin_edges

    def auto_correlogram(self, bin_width=2.0, width=202.0):
        """Compute the auto-correlogram.

        Arguments:
            bin_width: float (optional)
                The bin width of the auto-correlogram (ms).
                The default value is 2.0.
            width: float (optional)
                The width of the auto-correlogram (ms).
                The default value is 201.0.
        Returns:
            bin_counts: numpy.ndarray
            bin_edges: numpy.ndarray
        """

        bin_counts, bin_edges = self.cross_correlogram(self, bin_width=bin_width, width=width)

        bin_index = int(np.floor(0.5 * width / bin_width))
        bin_counts[bin_index] -= self.nb_times

        return bin_counts, bin_edges

    def interspike_interval_histogram(self, bin_width=0.5, width=25.0):
        """Compute the interspike interval histogram.

        Arguments:
            bin_width: float (optional)
                The bin width of the interspike interval histogram (ms).
                The default value is 0.5.
            width: float (optional)
                The width of the interspike interval histogram (ms).
                The default value is 25.0.
        Returns:
            bin_counts: numpy.ndarray
            bin_edges: numpy.ndarray
        """

        nb_bins = int(np.ceil(width / bin_width))
        bin_counts = np.zeros(nb_bins, dtype=np.int)

        times = np.sort(self.times)
        nb_times = times.size

        for index in range(0, nb_times - 1):
            interspike_interval_in_s = times[index + 1] - times[index + 0]
            interspike_interval_in_ms = interspike_interval_in_s * 1e+3
            if interspike_interval_in_ms < width:
                bin_index = int(np.floor(interspike_interval_in_ms / bin_width))
                bin_counts[bin_index] += 1

        bin_edges = np.linspace(0.0, width, num=nb_bins+1)

        return bin_counts, bin_edges

    def nb_refractory_period_violations(self, refractory_period=2.0):
        """Count the number of refractory period violations.

        Argument:
            refractory_period: float (optional)
                The refractory period used to distinguish violations (ms).
                The default value is 2.0.
        Return:
            nb_rpv: integer
                The number of refractory period violations.
        """

        nb_rpv = 0

        times = np.sort(self.times)
        nb_times = times.size

        for index in range(0, nb_times - 1):
            interspike_interval_in_s = times[index + 1] - times[index + 0]
            interspike_interval_in_ms = interspike_interval_in_s * 1e+3
            if interspike_interval_in_ms < refractory_period:
                nb_rpv += 1

        return nb_rpv

    def nb_rpv(self, refractory_period=2.0):
        """Alias for nb_refractory_period_violations."""

        return self.nb_refractory_period_violations(refractory_period=refractory_period)

    def refractory_period_violation_coefficient(self, refractory_period=2.0):
        """Compute the refractory period violations coefficient.

        Argument:
            refractory_period: float (optional)
                The refractory period used to distinguish violations (ms).
                The default value is 2.0.
        Return:
            rpv_coefficient: float
                The refractory period violation coefficient (i.e. number of violation divided by number of intervals).
        """

        nb_rpv = self.nb_refractory_period_violations(refractory_period=refractory_period)
        nb_intervals = self.nb_times - 1
        rpv_coefficient = float(nb_rpv) / float(nb_intervals)

        return rpv_coefficient

    def rpv_coefficient(self, refractory_period=2.0):
        """Alias for refractory_period_violation_coefficient."""

        return self.refractory_period_violation_coefficient(refractory_period=refractory_period)
