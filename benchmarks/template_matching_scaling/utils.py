# -*- coding: utf-8 -*-

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.sparse

from circusort import io
from circusort.io.synthetic import SyntheticStore
from circusort.io.template import TemplateStore
from circusort.block.synthetic_generator import Cell


class Results(object):
    """Results of the scenario"""
    # TODO complete docstring.

    def __init__(self, generator_kwargs, signal_writer_kwargs,
                 mad_writer_kwargs, peak_writer_kwargs,
                 updater_kwargs, spike_writer_kwargs):

        # Save raw input arguments.
        self.generator_kwargs = generator_kwargs
        self.signal_writer_kwargs = signal_writer_kwargs
        self.mad_writer_kwargs = mad_writer_kwargs
        self.peak_writer_kwargs = peak_writer_kwargs
        self.updater_kwargs = updater_kwargs
        self.spike_writer_kwargs = spike_writer_kwargs

        # Retrieve generation parameters.
        hdf5_path = os.path.abspath(self.generator_kwargs['hdf5_path'])
        self.gen = SyntheticStore(hdf5_path)
        log_path = os.path.abspath(self.generator_kwargs['log_path'])
        with open(log_path, mode='r') as log_file:
            self.generator = json.load(log_file)
        self.probe_path = self.generator_kwargs['probe']

        # Retrieve detected peak.
        peaks_path = self.peak_writer_kwargs['neg_peaks']
        self.detected_peaks = io.load_peaks(peaks_path)

        # Retrieve probe.
        self.probe = io.Probe(self.probe_path)

        # Sampling rate.
        self.sampling_rate = 20e+3  # [Hz]
        self.chunk_size = self.generator['nb_samples']

        # Retrieve detected spikes.
        spike_times_path = self.spike_writer_kwargs['spike_times']
        spike_templates_path = self.spike_writer_kwargs['templates']
        spike_amplitudes_path = self.spike_writer_kwargs['amplitudes']
        self.detected_spikes = io.load_spikes(spike_times_path,
                                              spike_templates_path,
                                              spike_amplitudes_path)

    @property
    def nb_channels(self):

        return self.probe.nb_channels

    # Voltage signal analysis.

    def plot_signal(self, t_min=None, t_max=None):
        """Plot signal"""

        # Retrieve signal data.
        path = self.signal_writer_kwargs['data_path']
        data = np.memmap(path, dtype=np.float32, mode='r')
        data = np.reshape(data, (-1, self.nb_channels))

        if t_min is None:
            i_min = 0
        else:
            i_min = int(t_min * self.sampling_rate)
        if t_max is None:
            i_max = data.shape[0]
        else:
            i_max = int(t_max * self.sampling_rate) + 1

        plt.figure()
        y_scale = 0.0
        for k in range(0, self.nb_channels):
            y = data[i_min:i_max, k]
            y_scale = max(y_scale, 2.0 * np.amax(np.abs(y)))
        for k in range(0, self.nb_channels):
            y = data[i_min:i_max, k]
            y_offset = float(k)
            x = np.arange(i_min, i_max).astype(np.float32) / self.sampling_rate
            plt.plot(x, y / y_scale + y_offset, c='C0')
        plt.xlabel("time (s)")
        plt.ylabel("electrode")
        plt.tight_layout()
        plt.show()

        return

    # Peak trains analysis.

    @property
    def generated_peak_train(self):

        generated_peak_train = self.gen.get(variables='spike_times')
        generated_peak_train = generated_peak_train[u'0']['spike_times']
        generated_peak_train = generated_peak_train.astype(np.float32)
        generated_peak_train /= self.sampling_rate

        return generated_peak_train

    @property
    def detected_peak_trains(self):

        detected_peak_trains = {}
        for k in range(0, self.nb_channels):
            detected_peak_train = self.detected_peaks.get_time_steps(k)
            detected_peak_train = detected_peak_train.astype(np.float32)
            detected_peak_train /= self.sampling_rate
            detected_peak_trains[k] = detected_peak_train

        return detected_peak_trains

    def compare_peak_trains(self):
        """Compare peak trains"""

        # Plot peak trains to compare them visually.
        plt.figure()
        # Plot generated peak train.
        x = [t for t in self.generated_peak_train]
        y = [0.0 for _ in x]
        plt.scatter(x, y, c='C1', marker='|')
        # Plot detected peak trains.
        detected_peak_trains = self.detected_peak_trains
        for k in range(0, self.nb_channels):
            x = [t for t in detected_peak_trains[k]]
            y = [float(k + 1) for _ in x]
            plt.scatter(x, y, c='C0', marker='|')
        plt.xlabel("time (arb. unit)")
        plt.ylabel("peak train")
        plt.title("Peak trains comparison")
        plt.tight_layout()
        plt.show()

        return

    def compare_peaks_number(self):
        """Compare number of peaks"""

        # Compute the number of generated peaks.
        nb_generated_peaks = self.generated_peak_train.size
        # Print the number of generated peaks.
        msg = "number of generated peaks: {}"
        print(msg.format(nb_generated_peaks))
        # Retrieve the detected peaks.
        detected_peak_trains = self.detected_peak_trains
        # Compute the number of detected peaks.
        nb_detected_peaks = 0
        for k in range(0, self.nb_channels):
            nb_detected_peaks += detected_peak_trains[k].size
        # Print the number of detected peaks.
        msg = "number of detected peaks: {}"
        print(msg.format(nb_detected_peaks))
        # Print the number of detected peaks per channel.
        for k in range(0, self.nb_channels):
            msg = "number of detected peaks on channel {}: {} [{:.1f}%]"
            p = 100.0 * float(detected_peak_trains[k].size) / float(nb_detected_peaks)
            print(msg.format(k, detected_peak_trains[k].size, p))

        return

    @staticmethod
    def compute_ipis(train, t_min=None, t_max=None):
        """Compute interpeak intervals"""

        train = np.sort(train)
        if t_min is not None and t_max is not None:
            assert t_min <= t_max
        if t_min is not None:
            train = train[train >= t_min]
        if t_max is not None:
            train = train[train <= t_max]
        ipis = train[+1:] - train[:-1]
        ipis = np.sort(ipis)

        return ipis

    def plot_cum_dist_ipis(self, train, t_min=None, t_max=None, d_min=0.0, d_max=200.0, ax=None, **kwargs):
        """Plot cumulative distribution of IPIs"""

        d_min = d_min * 1e-3  # ms
        d_max = d_max * 1e-3  # ms
        ipis = self.compute_ipis(train, t_min=t_min, t_max=t_max)
        y_min = np.sum(ipis <= d_min)
        ipis = ipis[d_min < ipis]
        ipis = ipis[ipis <= d_max]
        x = np.unique(ipis)
        y = np.array([y_min + np.sum(ipis <= e) for e in x])
        x = np.insert(x, 0, [d_min])
        y = np.insert(y, 0, [y_min])
        x = np.append(x, [d_max])
        y = np.append(y, y[-1])

        if ax is None:
            plt.style.use('seaborn-paper')
            plt.figure()
            ax = plt.gca()
            ax.set_xlabel("duration (ms)")
            ax.set_ylabel("number")
        ax.step(1e+3 * x, y, where='post', **kwargs)

        return

    def plot_cum_dists_ipis(self, t_min=None, t_max=None, d_min=0.0, d_max=200.0):
        """Plot cumulative distributions of IPIs

        Arguments:
            t_min: none | float (optional)
                Start time of each peak trains. The default value is None.
            t_max: none | float (optional)
                End time of each peak trains. The default value is None.
            d_min: float (optional)
                Minimal interpeak interval duration [ms]. The default value is 0.0.
            d_max: float (optional)
                Maximal interpeak interval duration [ms]. The default value is 200.0.
        """

        assert 0.0 <= d_min <= d_max

        plt.style.use('seaborn-paper')
        plt.figure()
        ax = plt.gca()
        self.plot_cum_dist_ipis(self.generated_peak_train, t_min=t_min, t_max=t_max, d_min=d_min, d_max=d_max,
                                ax=ax, c='C0', label='generated')
        for k in self.detected_peak_trains:
            c = 'C{}'.format((k % 9) + 1)
            label = 'detected {}'.format(k + 1)
            self.plot_cum_dist_ipis(self.detected_peak_trains[k], t_min=t_min, t_max=t_max, d_min=d_min, d_max=d_max,
                                    ax=ax, c=c, label=label)
        ax.set_xlabel("duration (ms)")
        ax.set_ylabel("number")
        ax.set_title("Cumulative distributions of IPIs")
        ax.legend()
        plt.show()

        return

    def plot_signal_and_peaks(self, t_min=None, t_max=None, thold=1.0):
        """Plot signal and peaks"""

        # Retrieve signal data.
        path = self.signal_writer_kwargs['data_path']
        data = np.memmap(path, dtype=np.float32, mode='r')
        data = np.reshape(data, (-1, self.nb_channels))

        # Retrieve threshold data.
        mad_path = self.mad_writer_kwargs['data_path']
        mad_data = np.memmap(mad_path, dtype=np.float32, mode='r')
        mad_data = np.reshape(mad_data, (-1, self.nb_channels))

        if t_min is None:
            i_min = 0
        else:
            i_min = int(t_min * self.sampling_rate)
        if t_max is None:
            i_max = data.shape[0]
        else:
            i_max = int(t_max * self.sampling_rate) + 1

        plt.figure()
        # Compute scaling factor.
        y_scale = 0.0
        for k in range(0, self.nb_channels):
            y = data[i_min:i_max, k]
            y_scale = max(y_scale, 2.0 * np.amax(np.abs(y)))
        # Plot electrode signals.
        for k in range(0, self.nb_channels):
            y = data[i_min:i_max, k]
            y_offset = float(k)
            x = np.arange(i_min, i_max).astype(np.float32) / self.sampling_rate
            plt.plot(x, y / y_scale + y_offset, c='C0', zorder=1)
        # Plot MADs.
        for k in range(0, self.nb_channels):
            mads = mad_data[:, k]
            i = np.arange(0, mads.size) * self.chunk_size
            x = i.astype(np.float32) / self.sampling_rate
            mask = np.array([t_min <= t <= t_max for t in x])
            x = x[mask]
            y = thold * mads[mask]
            y_offset = float(k)
            plt.step(x, + y / y_scale + y_offset, where='post', c='C3')
            plt.step(x, - y / y_scale + y_offset, where='post', c='C3')
        # Plot generated peaks.
        x = [t for t in self.generated_peak_train if t_min <= t <= t_max]
        y = [-1.0 for _ in x]
        plt.scatter(x, y, c='C2', marker='|', zorder=2)
        # Plot detected peaks.
        detected_peak_trains = self.detected_peak_trains
        for k in range(0, self.nb_channels):
            x = [t for t in detected_peak_trains[k] if t_min <= t <= t_max]
            y = [float(k) for _ in x]
            plt.scatter(x, y, c='C1', marker='|', zorder=2)
        plt.xlabel("time (s)")
        plt.ylabel("electrode")
        plt.tight_layout()
        plt.show()

        return

    # Spike trains analysis.

    @property
    def generated_spike_steps(self):

        generated_spike_steps = self.gen.get(variables='spike_times')
        generated_spike_steps = generated_spike_steps[u'0']['spike_times']
        generated_spike_steps = generated_spike_steps.astype(np.int32)

        return generated_spike_steps

    @property
    def generated_spike_train(self):

        generated_spike_train = self.generated_spike_steps
        generated_spike_train = generated_spike_train.astype(np.float32)
        generated_spike_train /= self.sampling_rate

        return generated_spike_train

    def get_detected_spike_trains(self, t_min=None, t_max=None):
        """Get detected spike trains

        Arguments:
            t_min: none | float (optional)
                Start time for the comparison (in s). The default value is None.
            t_max: none | float (optional)
                End time for the comparison (in s). The default value is None.
        """

        trains = {}
        for k in self.detected_spikes.units:
            train = self.detected_spikes.get_time_steps(k)
            train = train.astype(np.float32)
            train /= self.sampling_rate
            if t_min is not None:
                train = train[t_min <= train]
            if t_max is not None:
                train = train[train <= t_max]
            trains[k] = train

        return trains

    def get_generated_spike_trains(self, t_min=None, t_max=None):
        """Get generated spike trains

        Arguments:
            t_min: none | float (optional)
                Start time for the comparison (in s). The default value is None.
            t_max: none | float (optional)
                End time for the comparison (in s). The default value is None.
        """

        spike_times = self.gen.get(variables='spike_times')
        trains = {}
        for k in range(0, self.gen.nb_cells):
            key = u'{}'.format(k)
            train = spike_times[key]['spike_times']
            train = train.astype(np.float32)
            train /= self.sampling_rate
            if t_min is not None:
                train = train[t_min <= train]
            if t_max is not None:
                train = train[train <= t_max]
            trains[k] = train

        return trains

    def compare_spike_trains(self, t_min=None, t_max=None):
        """Compare spike trains

        Arguments:
            t_min: none | float (optional)
                Start time for the comparison (in s). The default value is None.
            t_max: none | float (optional)
                End time for the comparison (in s). The default value is None.
        """

        # Retrieve detected spike trains.
        detected_spike_trains = self.get_detected_spike_trains(t_min=t_min, t_max=t_max)
        # Retrieve generated spike trains.
        generated_spike_trains = self.get_generated_spike_trains(t_min=t_min, t_max=t_max)
        # Compute number of detected spike trains.
        nb_detected_spike_trains = len(detected_spike_trains)
        # Compute number of generated spike trains.
        nb_generated_spike_trains = len(generated_spike_trains)

        # Plot spike trains to compare them visually.
        plt.figure()
        # Plot generated spike trains.
        for k, train in generated_spike_trains.iteritems():
            x = [t for t in train]
            y = [float(k + 0) for _ in x]
            if k == 0:
                plt.scatter(x, y, c='C0', marker='|', label='generated')
            else:
                plt.scatter(x, y, c='C0', marker='|')
        # Plot detected spike trains.
        for k, train in detected_spike_trains.iteritems():
            x = [t for t in train]
            y = [float(k + nb_generated_spike_trains) for _ in x]
            if k == 0:
                plt.scatter(x, y, c='C1', marker='|', label='detected')
            else:
                plt.scatter(x, y, c='C1', marker='|')
        y = [v for v in range(0, nb_generated_spike_trains)]\
            + [v + nb_generated_spike_trains for v in range(0, nb_detected_spike_trains)]
        labels = ["{}".format(v) for v in range(0, nb_generated_spike_trains)]\
            + ["{}".format(v) for v in range(0, nb_detected_spike_trains)]
        plt.yticks(y, labels)
        plt.xlabel("time (s)")
        plt.ylabel("train")
        plt.title("Spike trains comparison")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return

    def compute_firing_rate(self, train, t_min=None, t_max=None, bin_width=1.0, **kwargs):
        """Compute the firing rate

        Arguments:
            train: np.ndarray
                Spike train.
            t_min: none | float
                Start time (in s). The default value is None.
            t_max: none | float
                End time (in s). The default value is None.
            bin_width: float (optional)
                Bin width (in s). The default value is 1.0.

        See also:
            numpy.histogram for additional keyword arguments.
        """

        if t_min is None:
            t_min = np.min(train) if train.size > 0 else 0.0
        if t_max is None:
            t_max = np.max(train) if train.size > 0 else 0.0
        nb_bins = int(np.ceil((t_max - t_min) / bin_width))
        bins = [t_min + float(i) * bin_width for i in range(0, nb_bins)]  # bin edges
        bin_values, bin_edges = np.histogram(train, bins=bins, **kwargs)
        rates = bin_values.astype(np.float) / bin_width

        return rates, bin_edges

    def get_detected_firing_rates(self, t_min=None, t_max=None, bin_width=1.0):

        trains = self.get_detected_spike_trains(t_min=t_min, t_max=t_max)
        rates, bin_edges = {}, None
        for k, train in trains.iteritems():
            rates[k], bin_edges = self.compute_firing_rate(train, t_min=t_min, t_max=t_max, bin_width=bin_width)

        return rates, bin_edges

    def get_generated_firing_rates(self, t_min=None, t_max=None, bin_width=1.0):

        trains = self.get_generated_spike_trains(t_min=t_min, t_max=t_max)
        rates, bin_edges = {}, None
        for k, train in trains.iteritems():
            rates[k], bin_edges = self.compute_firing_rate(train, t_min=t_min, t_max=t_max, bin_width=bin_width)

        return rates, bin_edges

    def inspect_firing_rates(self, **kwargs):
        """Firing rates inspection

        See also:
            get_detected_firing_rates for additional keyword arguments.
            get_generated_firing_rates for additional keyword arguments.
        """

        # Retrieve detected firing rates.
        detected_firing_rates, bin_edges = self.get_detected_firing_rates(**kwargs)
        # Retrieve generated firing rates.
        generated_firing_rates, bin_edges = self.get_generated_firing_rates(**kwargs)
        # # Compute number of detected spike trains.
        # nb_detected_spike_trains = len(detected_spike_trains)
        # # Compute number of generated spike trains.
        # nb_generated_spike_trains = len(generated_spike_trains)

        # Plot firing rates to compare them visually.
        plt.style.use('seaborn-paper')
        plt.figure()
        # Plot detected firing rates.
        for k, rate in detected_firing_rates.iteritems():
            x = bin_edges[:-1]
            y = rate
            if k == 0:
                plt.plot(x, y, c='C1', label='detected')
            else:
                plt.plot(x, y, c='C1')
        # Plot generated firing rates.
        for k, rate in generated_firing_rates.iteritems():
            x = bin_edges[:-1]
            y = rate
            if k == 0:
                plt.plot(x, y, c='C0', label='generated')
            else:
                plt.plot(x, y, c='C0')
        plt.xlabel("time (s)")
        plt.ylabel("rate (Hz)")
        plt.title("Firing rate comparison")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return

    def van_rossum_distances(self, t_min=None, t_max=None, c=100.0):
        """Compute von Rossum distance between generated and detected spike trains"""

        # Retrieve the generated spike train.
        generated_spike_train = self.generated_spike_train

        # Retrieve the detected spike trains.
        detected_spike_trains = self.get_detected_spike_trains()

        d = np.zeros(len(detected_spike_trains))
        for k, train in enumerate(detected_spike_trains.values()):
            # Initialize distance.
            d[k] = 0.0
            # Collect spike times and polarities.
            nb_spikes = len(generated_spike_train) + len(train)
            t = np.zeros(nb_spikes, dtype=float)
            p = np.zeros(nb_spikes, dtype=float)
            i_gen = 0
            i_det = 0
            for i in range(0, nb_spikes):
                if i_gen == len(generated_spike_train):
                    t[i] = train[i_det]
                    p[i] = -1.0
                    i_det += 1
                elif i_det == len(train):
                    t[i] = generated_spike_train[i_gen]
                    p[i] = +1.0
                    i_gen += 1
                elif generated_spike_train[i_gen] < train[i_det]:
                    t[i] = generated_spike_train[i_gen]
                    p[i] = +1.0
                    i_gen += 1
                else:
                    t[i] = train[i_det]
                    p[i] = -1.0
                    i_det += 1
            # Keep spike times between t_min and t_max.
            if t_min is not None:
                mask = t_min <= t
                t = t[mask]
                p = p[mask]
            if t_max is not None:
                mask = t <= t_max
                t = t[mask]
                p = p[mask]
            nb_spikes = t.size
            # Add area between (i-1)th and (i)th spikes.
            for i in range(1, nb_spikes):
                dd = 0.0
                for j in range(0, i):
                    a = t[i - 1] - t[j]
                    b = t[i] - t[j]
                    da = c * (np.exp(- a / c) - np.exp(- b / c))
                    dd += p[j] * da
                dd = np.abs(dd)
                d[k] += dd
            # Add area after last spike if it exists.
            if nb_spikes > 0:
                dd = 0.0
                for j in range(0, nb_spikes):
                    a = t[-1] - t[j]
                    da = c * np.exp(- a / c)
                    dd += p[j] * da
                dd = np.abs(dd)
                d[k] += dd
            d[k] /= float(nb_spikes) * c

        return d

    @staticmethod
    def compute_isis(train, t_min=None, t_max=None):
        """Compute interspike intervals"""

        train = np.sort(train)
        if t_min is not None and t_max is not None:
            assert t_min <= t_max
        if t_min is not None:
            train = train[train >= t_min]
        if t_max is not None:
            train = train[train <= t_max]
        isis = train[+1:] - train[:-1]
        isis = np.sort(isis)

        return isis

    def plot_cum_dist_isis(self, train, t_min=None, t_max=None, d_min=0.0, d_max=200.0, ax=None, **kwargs):
        """Plot cumulative distribution of ISIs"""

        d_min = d_min * 1e-3  # ms
        d_max = d_max * 1e-3  # ms
        isis = self.compute_isis(train, t_min=t_min, t_max=t_max)
        y_min = np.sum(isis <= d_min)
        isis = isis[d_min < isis]
        isis = isis[isis <= d_max]
        x = np.unique(isis)
        y = np.array(y_min + [np.sum(isis <= e) for e in x])
        x = np.insert(x, 0, [d_min])
        y = np.insert(y, 0, [y_min])
        x = np.append(x, [d_max])
        y = np.append(y, y[-1])

        if ax is None:
            plt.style.use('seaborn-paper')
            plt.figure()
            ax = plt.gca()
            ax.set_xlabel("duration (ms)")
            ax.set_ylabel("number")
        ax.step(1e+3 * x, y, where='post', **kwargs)

        return

    def plot_cum_dists_isis(self, detected_units=None, generated_units=None,
                            t_min=None, t_max=None, d_min=0.0, d_max=200.0, ax=None):
        """Plot cumulative distributions of ISIs

        Arguments:
            detected_units: none | iterable (optional)
                Detected units. The default value is None.
            generated_units: none | iterable (optional)
                Generated units. The default value is None.
            t_min: none | float (optional)
                Start time of each spike trains. The default value is None.
            t_max: none | float (optional)
                End time of each spike trains. The default value is None.
            d_min: float (optional)
                Minimal interspike interval duration [ms]. The default value is 0.0.
            d_max: float (optional)
                Maximal interspike interval duration [ms]. The default value is 200.0.
            ax: none | matplotlib.axes.Axes (optional)
                Matplotlib axes. The default value is None.
        """

        assert 0.0 <= d_min <= d_max

        if ax is None:
            plt.style.use('seaborn-paper')
            plt.figure()
            ax = plt.gca()
            is_subplot = False
        else:
            is_subplot = True
        generated_spike_trains = self.get_generated_spike_trains()
        if generated_units is not None:
            generated_spike_trains = {i: generated_spike_trains[i] for i in generated_units}
        for k, (i, train) in enumerate(generated_spike_trains.iteritems()):
            c = 'C{}'.format(2 * (k % 5) + 0)
            label = 'generated {}'.format(i)
            self.plot_cum_dist_isis(train, t_min=t_min, t_max=t_max, d_min=d_min, d_max=d_max, ax=ax, c=c, label=label)
        detected_spike_trains = self.get_detected_spike_trains()
        if detected_units is not None:
            detected_spike_trains = {i: detected_spike_trains[i] for i in detected_units}
        for k, (i, train) in enumerate(detected_spike_trains.iteritems()):
            c = 'C{}'.format(2 * (k % 5) + 1)
            label = 'detected {}'.format(i)
            self.plot_cum_dist_isis(train, t_min=t_min, t_max=t_max, d_min=d_min, d_max=d_max, ax=ax, c=c, label=label)
        if not is_subplot:
            ax.set_xlabel("duration (ms)")
            ax.set_ylabel("number of intervals")
            ax.set_title("Cumulative distributions of ISIs")
            ax.legend()
            plt.show()
        else:
            ax.set_xlabel("duration (ms)")

        return

    def plot_all_cum_dists_isis(self, matching=None, **kwargs):
        """Plot cumulative distributions of ISIs for a given matching

        Arguments:
            matching: none | list (optional)
                Matching between detected and generated units. The default value is None.

        See also:
            plot_cum_dists_isis
        """

        if matching is None:
            self.plot_cum_dist_isis(**kwargs)
        else:
            plt.style.use('seaborn-paper')
            nb_pairs = len(matching)
            _, ax_arr = plt.subplots(nrows=1, ncols=nb_pairs, sharey='row')
            for k, (detected_unit, generated_unit) in enumerate(matching):
                ax = ax_arr[k]
                self.plot_cum_dists_isis([detected_unit], [generated_unit], ax=ax, **kwargs)
                if k == 0:
                    ax.set_ylabel("number of intervals")
                ax.set_title("det. {} - gen. {}".format(detected_unit, generated_unit))
            plt.suptitle("Cumulative distributions of ISIs")
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            plt.show()

        return

    def plot_signal_and_spikes(self, t_min=None, t_max=None, thold=1.0):
        """Plot signal and spikes

        Arguments:
            t_min: float
            t_max: float
            thold: float

        """

        # Retrieve signal data.
        path = self.signal_writer_kwargs['data_path']
        data = np.memmap(path, dtype=np.float32, mode='r')
        data = np.reshape(data, (-1, self.nb_channels))

        # Retrieve threshold data.
        mad_path = self.mad_writer_kwargs['data_path']
        mad_data = np.memmap(mad_path, dtype=np.float32, mode='r')
        mad_data = np.reshape(mad_data, (-1, self.nb_channels))

        if t_min is None:
            i_min = 0
        else:
            i_min = int(t_min * self.sampling_rate)
        if t_max is None:
            i_max = data.shape[0]
        else:
            i_max = int(t_max * self.sampling_rate) + 1

        plt.figure()
        # Compute scaling factor.
        y_scale = 0.0
        for k in range(0, self.nb_channels):
            y = data[i_min:i_max, k]
            y_scale = max(y_scale, 2.0 * np.amax(np.abs(y)))
        # Plot electrode signals.
        for k in range(0, self.nb_channels):
            y = data[i_min:i_max, k]
            y_offset = float(k)
            x = np.arange(i_min, i_max).astype(np.float32) / self.sampling_rate
            plt.plot(x, y / y_scale + y_offset, c='C0', zorder=1)
        # Plot MADs.
        for k in range(0, self.nb_channels):
            mads = mad_data[:, k]
            i = np.arange(0, mads.size) * self.chunk_size
            x = i.astype(np.float32) / self.sampling_rate
            mask = np.array([t_min <= t <= t_max for t in x])
            x = x[mask]
            y = thold * mads[mask]
            y_offset = float(k)
            plt.step(x, + y / y_scale + y_offset, where='post', c='C3')
            plt.step(x, - y / y_scale + y_offset, where='post', c='C3')
        # Plot generated spike train.
        x = [t for t in self.generated_spike_train if t_min <= t <= t_max]
        y = [-1.0 for _ in x]
        plt.scatter(x, y, c='C2', marker='|', zorder=2)
        # Plot detected spike trains.
        detected_spike_trains = self.get_detected_spike_trains()
        for k, train in enumerate(detected_spike_trains.values()):
            x = [t for t in train if t_min <= t <= t_max]
            y = [-float(k + 2) for _ in x]
            plt.scatter(x, y, c='C1', marker='|', zorder=2)
        plt.xlabel("time (s)")
        plt.ylabel("electrode")
        plt.ylabel("Signal and spikes")
        plt.tight_layout()
        plt.show()

        return

    # Comparison between generated and fitted template.

    @property
    def nb_detected_units(self):

        template_path = os.path.abspath(self.updater_kwargs['data_path'])
        template_store = TemplateStore(template_path, mode='r')
        nb_detected_units = template_store.nb_templates
        template_store.close()

        return nb_detected_units

    @property
    def nb_generated_units(self):

        nb_generated_units = self.gen.nb_cells

        return nb_generated_units

    def get_generated_template(self, i, time=None, nn=100, hf_dist=45, a_dist=1.0):
        """Get generated templates
        
        Arguments:
            i: int
                Cell identifier.
            time: None (optional)
            nn: float (optional)
            hf_dist: float (optional)
            a_dist: float (optional)
        """

        if time is None:
            time = 0
        res = self.gen.get(indices=[i], variables=['x', 'y', 'z'])
        cell = Cell(lambda t: res[i]['x'][time],
                    lambda t: res[i]['y'][time],
                    lambda t: res[i]['z'][time],
                    nn=nn, hf_dist=hf_dist, a_dist=a_dist)
        a, b, c = cell.get_waveforms(time, self.probe)
        template = scipy.sparse.csc_matrix((c, (b, a + 20)), shape=(self.nb_channels, 81))
        template = template.toarray()

        return template

    def get_detected_template(self, i):
        """Get detected templates

        Arguments:
            i: int
                Unit identifier.
        """

        template_path = os.path.abspath(self.updater_kwargs['data_path'])
        template_store = TemplateStore(template_path, mode='r')

        data = template_store.get([i], ['templates', 'norms'])
        width = template_store.width
        templates = data.pop('templates').T
        norms = data.pop('norms')
        templates = templates[0].toarray()
        templates = np.reshape(templates, (self.nb_channels, width))
        templates *= norms[0]

        template_store.close()

        return templates

    def get_detected_spike_train(self, i):

        train = self.detected_spikes.get_time_steps(i)
        train = train.astype(np.int32)

        return train

    def get_generated_spike_train(self, i):

        key = u'{}'.format(i)
        train = self.gen.get(variables='spike_times')
        train = train[key]['spike_times']
        train = train.astype(np.int32)

        return train

    def get_sta(self, trains, time_window=5.0, time_shift=0.4, dtype=np.float32):
        """Get spike triggered average

        Arguments:
            trains: np.ndarray
                Spike trains (i.e. array of spike time steps).
            time_window: float (optional)
                Time window [ms]. The default value is 5.0.
            time_shift: float (optional)
                Time shift [ms]. The default value is 1.0.
            dtype: type (optional)
                Data type. The default value is np.float32.
        """

        # Retrieve the signal data.
        path = self.signal_writer_kwargs['data_path']
        data = np.memmap(path, dtype=dtype, mode='r')
        data = np.reshape(data, (-1, self.nb_channels))

        # Initialize averaged template.
        nb_samples = int(time_window * 1e-3 * self.sampling_rate)
        i_shift = int(time_shift * 1e-3 * self.sampling_rate)
        di = nb_samples // 2
        template = np.zeros((nb_samples, self.nb_channels), dtype=dtype)
        count = 0
        for i in trains:
            i_ = i + i_shift
            i_min = i_ - di
            i_max = i_ + di + 1
            spike_data = data[i_min:i_max, :]
            if spike_data.shape[0] != i_max - i_min:
                pass
            elif count == 0:
                template = spike_data
                count = 1
            else:
                template = (float(count - 1) / float(count)) * template + (1.0 / float(count)) * spike_data
                count += 1
        template = np.transpose(template)

        return template

    def compare_templates(self, ij, time_shift=0.4, ax=None):
        """Compare templates of one generated unit with one detected unit.

        Arguments:
            ij: tuple
                Pair of indices. The first one is the index of the detected unit of interest. The second one is the
                index of the generated unit of interest, e.g. (0, 0).
            time_shift: float (optional)
                Time shift [ms]. The default value is 0.4.
            ax: none | matplotlib.axes.Axes (optional)
        """

        assert 0 <= ij[0] < self.nb_detected_units,\
            "detected unit {} does not exist, {} units were detected".format(ij[0], self.nb_detected_units)
        assert 0 <= ij[1] < self.nb_generated_units,\
            "generated unit {} does not exist, {} units were generated".format(ij[1], self.nb_generated_units)

        # Retrieve detected template.
        detected_template = self.get_detected_template(ij[0])
        # Retrieve generated template.
        generated_template = self.get_generated_template(ij[1])
        # Retrieve STA of detected spike train.
        detected_spike_train = self.get_detected_spike_train(ij[0])
        detected_sta = self.get_sta(detected_spike_train, time_shift=time_shift)
        # Retrieve STA of generated spike train.
        generated_spike_train = self.get_generated_spike_train(ij[1])
        generated_sta = self.get_sta(generated_spike_train, time_shift=time_shift)

        scl = 0.9 * (self.probe.field_of_view['d'] / 2.0)
        alpha = 1.0
        if ax is None:
            _, ax = plt.subplots()
            is_subplot = False
        else:
            is_subplot = True
        # Plot the generated template.
        x_scl = scl
        y_scl = scl * (1.0 / np.max(np.abs(generated_template)))
        width = generated_template.shape[1]
        color = 'C0'
        for k in range(0, self.nb_channels):
            x_prb, y_prb = self.probe.positions[:, k]
            x = x_prb + x_scl * np.linspace(-1.0, +1.0, num=width)
            y = y_prb + y_scl * generated_template[k, :]
            if k == 0:
                ax.plot(x, y, c=color, alpha=alpha, label='generated template')
            else:
                ax.plot(x, y, c=color, alpha=alpha)
        # Plot the generated STA.
        x_scl = scl
        if np.max(np.abs(generated_sta)) == 0.0:
            y_scl = scl
        else:
            y_scl = scl * (1.0 / np.max(np.abs(generated_sta)))
        width = generated_sta.shape[1]
        color = 'C0'
        for k in range(0, self.nb_channels):
            x_prb, y_prb = self.probe.positions[:, k]
            x = x_prb + x_scl * np.linspace(-1.0, +1.0, num=width)
            y = y_prb + y_scl * generated_sta[k, :]
            if k == 0:
                ax.plot(x, y, c=color, alpha=alpha, linestyle='--', label='generated STA')
            else:
                ax.plot(x, y, c=color, alpha=alpha, linestyle='--')
        # Plot the detected template.
        x_scl = scl
        y_scl = scl * (1.0 / np.max(np.abs(detected_template)))
        width = detected_template.shape[1]
        color = 'C1'
        for k in range(0, self.nb_channels):
            x_prb, y_prb = self.probe.positions[:, k]
            x = x_prb + x_scl * np.linspace(-1.0, +1.0, num=width)
            y = y_prb + y_scl * detected_template[k, :]
            if k == 0:
                ax.plot(x, y, c=color, alpha=alpha, label='detected template')
            else:
                ax.plot(x, y, c=color, alpha=alpha)
        # Plot the detected STA.
        x_scl = scl
        if np.max(np.abs(detected_sta)) == 0.0:
            y_scl = scl
        else:
            y_scl = scl * (1.0 / np.max(np.abs(detected_sta)))
        width = detected_sta.shape[1]
        color = 'C1'
        for k in range(0, self.nb_channels):
            x_prb, y_prb = self.probe.positions[:, k]
            x = x_prb + x_scl * np.linspace(-1.0, +1.0, num=width)
            y = y_prb + y_scl * detected_sta[k, :]
            if k == 0:
                ax.plot(x, y, c=color, alpha=alpha, linestyle='--', label='detected STA')
            else:
                ax.plot(x, y, c=color, alpha=alpha, linestyle='--')
        if not is_subplot:
            plt.xlabel("x (µm)")
            plt.ylabel("y (µm)")
            plt.title("Templates comparison")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return

    def compare_all_templates(self):
        """Compare templates of generated units with detected units."""

        nb_rows = self.nb_generated_units
        nb_cols = self.nb_detected_units
        _, ax_arr = plt.subplots(nb_rows, nb_cols, sharex='all', sharey='all')
        for row in range(0, nb_rows):
            for col in range(0, nb_cols):
                self.compare_templates((col, row), ax=ax_arr[row, col])
        # Fine-tune figure.
        for col in range(0, nb_cols):
            ax_arr[-1, col].set_xlabel("detected {}".format(col))
        for row in range(0, nb_rows):
            ax_arr[row, 0].set_ylabel("generated {}".format(row))
        for row in range(0, nb_rows - 1):
            plt.setp([a.get_xticklabels() for a in ax_arr[row, :]], visible=False)
            plt.setp([a.get_xticklines() for a in ax_arr[row, :]], visible=False)
        for col in range(1, nb_cols):
            plt.setp([a.get_yticklabels() for a in ax_arr[:, col]], visible=False)
            plt.setp([a.get_yticklines() for a in ax_arr[:, col]], visible=False)
        plt.suptitle("Templates comparison")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, wspace=0.0, hspace=0.0)
        plt.show()

        return
