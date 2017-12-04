# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import json
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
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
        self.probe = io.load_probe(self.probe_path)

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

    def plot_synthetic_spatial_layout(self):

        radius = 5
        positions = self.probe.positions

        cells_radius = 7
        cells_positions = self.gen.get(variables=['x', 'y'])

        x_min = np.min(positions[0, :]) - float(radius)
        x_max = np.max(positions[0, :]) + float(radius)
        x_pad = 0.2 * (x_max - x_min)
        y_min = np.min(positions[1, :]) - float(radius)
        y_max = np.max(positions[1, :]) + float(radius)
        y_pad = 0.2 * (x_max - x_min)

        plt.style.use('seaborn-paper')

        plt.figure()
        ax = plt.gca()
        # Plot the tips of the electrodes.
        for k in range(self.probe.nb_channels):
            x, y = positions[:, k]
            c = plt.Circle((x, y), radius=radius, facecolor='gray', edgecolor='black')
            ax.add_artist(c)
        # Plot the cells.
        for k in range(0, self.gen.nb_cells):
            x, y = [cells_positions[str(k)][key][0] for key in ['x', 'y']]
            c = plt.Circle((x, y), radius=cells_radius, color='C{}'.format(k))
            ax.add_artist(c)
        plt.axis('scaled')
        plt.xlim(x_min - x_pad, x_max + x_pad)
        plt.ylim(y_min - y_pad, y_max + y_pad)
        plt.xlabel("x (µm)")
        plt.ylabel("y (µm)")
        plt.title("Synthetic spatial layout")
        plt.tight_layout()
        plt.show()

        return

    def plot_synthetic_templates(self):

        radius = 5
        positions = self.probe.positions

        x_min = np.min(positions[0, :]) - float(radius)
        x_max = np.max(positions[0, :]) + float(radius)
        x_pad = 0.2 * (x_max - x_min)
        y_min = np.min(positions[1, :]) - float(radius)
        y_max = np.max(positions[1, :]) + float(radius)
        y_pad = 0.2 * (x_max - x_min)

        scl = 0.9 * (self.probe.field_of_view['d'] / 2.0)
        alpha = 1.0

        plt.style.use('seaborn-paper')

        plt.figure()
        # Plot the generated template.
        for i in range(0, self.gen.nb_cells):
            # Retrieve the generated template.
            generated_template = self.generated_templates(i)
            x_scl = scl
            y_scl = scl * (1.0 / np.max(np.abs(generated_template)))
            width = generated_template.shape[1]
            color = 'C{}'.format(i)
            for k in range(self.nb_channels):
                x_prb, y_prb = self.probe.positions[:, k]
                x = x_prb + x_scl * np.linspace(-1.0, +1.0, num=width)
                y = y_prb + y_scl * generated_template[k, :]
                plt.plot(x, y, c=color, alpha=alpha)
        plt.axis('scaled')
        plt.xlim(x_min - x_pad, x_max + x_pad)
        plt.ylim(y_min - y_pad, y_max + y_pad)
        plt.xlabel("x (µm)")
        plt.ylabel("y (µm)")
        plt.title("Synthetic template")
        plt.tight_layout()
        plt.show()

        return

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
            msg = "number of detected peaks on channel {}: {} [{:2f}%]"
            p = 100.0 * float(detected_peak_trains[k].size) / float(nb_detected_peaks)
            print(msg.format(k, detected_peak_trains[k].size), p)

        return

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

        plt.style.use('seaborn-paper')

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
        plt.ylabel("channel")
        plt.title("Synthetic voltage signal (filtered)")
        plt.tight_layout()
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

    @property
    def detected_spike_trains(self):

        detected_spike_trains = {}
        for k in self.detected_spikes.units:
            detected_spike_train = self.detected_spikes.get_time_steps(k)
            detected_spike_train = detected_spike_train.astype(np.float32)
            detected_spike_train /= self.sampling_rate
            detected_spike_trains[k] = detected_spike_train

        return detected_spike_trains

    def compare_spike_trains(self):
        """Compare spike trains."""

        # Retrieve the generated spike train.
        generated_spike_train = self.generated_spike_train

        # Retrieve the inferred spike trains.
        detected_spike_trains = self.detected_spike_trains

        # Plot spike trains to compare them visually.
        plt.figure()
        # Plot generated spike train.
        x = [t for t in generated_spike_train]
        y = [0.0 for _ in x]
        plt.scatter(x, y, c='C1', marker='|')
        # Plot detected spike trains.
        for k, train in enumerate(detected_spike_trains.values()):
            x = [t for t in train]
            y = [float(k + 1) for _ in x]
            plt.scatter(x, y, c='C0', marker='|')
        plt.xlabel("time (s)")
        plt.ylabel("spike train")
        plt.title("Spike trains comparison")
        plt.tight_layout()
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
        detected_spike_trains = self.detected_spike_trains
        for k, train in enumerate(detected_spike_trains.values()):
            x = [t for t in train if t_min <= t <= t_max]
            y = [float(k) for _ in x]
            plt.scatter(x, y, c='C1', marker='|', zorder=2)
        plt.xlabel("time (s)")
        plt.ylabel("electrode")
        plt.ylabel("Signal and spikes")
        plt.tight_layout()
        plt.show()

        return

    # Comparison between generated and fitted template.

    def generated_templates(self, i, time=None, nn=100, hf_dist=45,
                            a_dist=1.0):
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
                    lambda t: res[i]['z'][time], nn=nn,
                    hf_dist=hf_dist, a_dist=a_dist)
        a, b, c = cell.get_waveforms(time, self.probe)
        template = scipy.sparse.csc_matrix((c, (b, a + 20)),
                                           shape=(self.nb_channels, 81))

        template = template.toarray()

        return template

    def detected_templates(self, i):
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

        return templates

    def averaged_template(self, time_window=5.0, time_shift=0.4,
                          dtype=np.float32):
        """Get averaged template
        Arguments:
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

        # Retrieve the generated spike times.
        generated_spike_steps = self.generated_spike_steps

        # Initialize averaged template.
        template = None
        count = 0
        nb_samples = int(time_window * 1e-3 * self.sampling_rate)
        i_shift = int(time_shift * 1e-3 * self.sampling_rate)
        di = nb_samples // 2
        for i in generated_spike_steps:
            i_ = i + i_shift
            i_min = i_ - di
            i_max = i_ + di + 1
            spike_data = data[i_min:i_max, :]
            if spike_data.shape[0] != i_max - i_min:
                pass
            elif template is None:
                template = spike_data
                count = 1
            else:
                template = (float(count - 1) / float(count)) * template\
                           + (1.0 / float(count)) * spike_data
                count += 1
        template = np.transpose(template)

        return template

    def compare_templates(self, time_shift=0.4):
        """Compare templates

        Compare the generated template with the detected template and the
        averaged template.

        Attribute:
            time_shift: float (optional)
                Time shift [ms]. The default value is 0.4.
        """

        # Retrieve the generated template.
        generated_template = self.generated_templates(0)

        # Retrieve the detected template.
        detected_template = self.detected_templates(0)

        # Retrieve the averaged template.
        averaged_template = self.averaged_template(time_shift=time_shift)

        scl = 0.9 * (self.probe.field_of_view['d'] / 2.0)
        alpha = 1.0
        plt.figure()
        # Plot the generated template.
        x_scl = scl
        y_scl = scl * (1.0 / np.max(np.abs(generated_template)))
        width = generated_template.shape[1]
        color = 'C0'
        for k in range(self.nb_channels):
            x_prb, y_prb = self.probe.positions[:, k]
            x = x_prb + x_scl * np.linspace(-1.0, +1.0, num=width)
            y = y_prb + y_scl * generated_template[k, :]
            if k == 0:
                plt.plot(x, y, c=color, alpha=alpha, label='generated')
            else:
                plt.plot(x, y, c=color, alpha=alpha)
        # Plot the detected template.
        x_scl = scl
        y_scl = scl * (1.0 / np.max(np.abs(detected_template)))
        width = detected_template.shape[1]
        color = 'C1'
        for k in range(self.nb_channels):
            x_prb, y_prb = self.probe.positions[:, k]
            x = x_prb + x_scl * np.linspace(-1.0, +1.0, num=width)
            y = y_prb + y_scl * detected_template[k, :]
            if k == 0:
                plt.plot(x, y, c=color, alpha=alpha, label='detected')
            else:
                plt.plot(x, y, c=color, alpha=alpha)
        # Plot the averaged template.
        x_scl = scl
        y_scl = scl * (1.0 / np.max(np.abs(averaged_template)))
        width = averaged_template.shape[1]
        color = 'C2'
        for k in range(self.nb_channels):
            x_prb, y_prb = self.probe.positions[:, k]
            x = x_prb + x_scl * np.linspace(-1.0, +1.0, num=width)
            y = y_prb + y_scl * averaged_template[k, :]
            if k == 0:
                plt.plot(x, y, c=color, alpha=alpha, label='averaged')
            else:
                plt.plot(x, y, c=color, alpha=alpha)
        plt.xlabel("x [um]")
        plt.ylabel("y [um]")
        plt.title("Templates comparison")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return
