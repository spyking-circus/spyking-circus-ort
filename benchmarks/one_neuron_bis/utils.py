import matplotlib.pyplot as plt
import numpy as np
import os

from circusort import io
from circusort.io.synthetic import SyntheticStore


class Results(object):
    """Results of the scenario"""
    # TODO complete docstring.

    def __init__(self, generator, signal_writer, mad_writer, peak_writer, probe_path):

        # Save raw input arguments.
        self.generator = generator
        self.signal_writer = signal_writer
        self.mad_writer = mad_writer
        self.peak_writer = peak_writer
        self.probe_path = probe_path

        # Retrieve generated peaks.
        gen_path = os.path.abspath(self.generator.hdf5_path)
        self.gen = SyntheticStore(gen_path)

        # Retrieve detected peak.
        peaks_path = self.peak_writer.recorded_peaks['negative']
        self.detected_peaks = io.load_peaks(peaks_path)

        # Retrieve probe.
        self.probe = io.Probe(self.probe_path)

        # Sampling rate.
        self.sampling_rate = 20.0e+3  # [Hz]
        self.chunk_size = self.generator.nb_samples

    @property
    def nb_electrodes(self):

        return self.probe.nb_channels

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
        for k in range(0, self.nb_electrodes):
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
        for k in range(0, self.nb_electrodes):
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

        print("number of generated peaks: {}".format(self.generated_peak_train.size))
        detected_peak_trains = self.detected_peak_trains
        for k in range(0, self.nb_electrodes):
            print("number of detected peaks on electrode {}: {}".format(k, detected_peak_trains[k].size))

        return

    def plot_signal(self, t_min=None, t_max=None):
        """Plot signal"""

        # Retrieve signal data.
        path = self.signal_writer.data_path
        data = np.memmap(path, dtype=np.float32, mode='r')
        data = np.reshape(data, (-1, self.nb_electrodes))

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
        for k in range(0, self.nb_electrodes):
            y = data[i_min:i_max, k]
            y_scale = max(y_scale, 2.0 * np.amax(np.abs(y)))
        for k in range(0, self.nb_electrodes):
            y = data[i_min:i_max, k]
            y_offset = float(k)
            x = np.arange(i_min, i_max).astype(np.float32) / self.sampling_rate
            plt.plot(x, y / y_scale + y_offset, c='C0')
        plt.xlabel("time (s)")
        plt.ylabel("electrode")
        plt.tight_layout()
        plt.show()

        return

    def plot_signal_and_peaks(self, t_min=None, t_max=None, thold=1.0):
        """Plot signal and peaks"""

        # Retrieve signal data.
        path = self.signal_writer.data_path
        data = np.memmap(path, dtype=np.float32, mode='r')
        data = np.reshape(data, (-1, self.nb_electrodes))

        # Retrieve threshold data.
        mad_path = self.mad_writer.data_path
        mad_data = np.memmap(mad_path, dtype=np.float32, mode='r')
        mad_data = np.reshape(mad_data, (-1, self.nb_electrodes))

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
        for k in range(0, self.nb_electrodes):
            y = data[i_min:i_max, k]
            y_scale = max(y_scale, 2.0 * np.amax(np.abs(y)))
        # Plot electrode signals.
        for k in range(0, self.nb_electrodes):
            y = data[i_min:i_max, k]
            y_offset = float(k)
            x = np.arange(i_min, i_max).astype(np.float32) / self.sampling_rate
            plt.plot(x, y / y_scale + y_offset, c='C0', zorder=1)
        # Plot MADs.
        for k in range(0, self.nb_electrodes):
            mads = mad_data[:, k]
            i = np.arange(0, mads.size) * self.chunk_size
            x = i.astype(np.float32) / self.sampling_rate
            mask = np.array([t_min <= t and t <= t_max for t in x])
            x = x[mask]
            y = thold * mads[mask]
            y_offset = float(k)
            plt.step(x, + y / y_scale + y_offset, where='post', c='C3')
            plt.step(x, - y / y_scale + y_offset, where='post', c='C3')
        # Plot generated peaks.
        x = [t for t in self.generated_peak_train if t_min <= t and t <= t_max]
        y = [-1.0 for _ in x]
        plt.scatter(x, y, c='C2', marker='|', zorder=2)
        # Plot detected peaks.
        detected_peak_trains = self.detected_peak_trains
        for k in range(0, self.nb_electrodes):
            x = [t for t in detected_peak_trains[k] if t_min <= t and t <= t_max]
            y = [float(k) for _ in x]
            plt.scatter(x, y, c='C1', marker='|', zorder=2)
        plt.xlabel("time (s)")
        plt.ylabel("electrode")
        plt.tight_layout()
        plt.show()

        return
