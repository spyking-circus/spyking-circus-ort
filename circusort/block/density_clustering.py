from .block import Block
import numpy as np
import scipy.interpolate
import os
import shutil

from circusort.io.probe import load_probe
from circusort.utils.clustering import OnlineManager


__classname__ = 'DensityClustering'


class DensityClustering(Block):
    """Density clustering

    Inputs:
        data
        pcs
        peaks
        mads

    Output:
        templates

    """

    name = "Density Clustering"

    params = {
        'threshold_factor': 7.0,
        'alignment': True,
        'sampling_rate': 20000.,
        'spike_width': 5,
        'nb_waveforms': 10000,
        'channels': None,
        'probe_path': None,
        'radius': None,
        'm_ratio': 0.01,
        'noise_thr': 0.8,
        'n_min': 0.01,
        'dispersion': [5, 5],
        'sub_dim': 5,
        'extraction': 'median-raw',
        'two_components': False,
        'decay_factor': 0.01,
        'mu': 4.,
        'epsilon': 'auto',
        'theta': -np.log(0.001),
        'tracking': False,
        'safety_time': 'auto',
        'compression': 0.5,
        'local_merges': 3,
        'debug_plots': None,
        'debug_data': None
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        # The following lines are useful to avoid some PyCharm's warnings.
        self.threshold_factor = self.threshold_factor
        self.alignment = self.alignment
        self.sampling_rate = self.sampling_rate
        self.spike_width = self.spike_width
        self.nb_waveforms = self.nb_waveforms
        self.channels = self.channels
        self.probe_path = self.probe_path
        self.radius = self.radius
        self.m_ratio = self.m_ratio
        self.noise_thr = self.noise_thr
        self.n_min = self.n_min
        self.dispersion = self.dispersion
        self.two_components = self.two_components
        self.decay_factor = self.decay_factor / float(self.sampling_rate)
        self.mu = self.mu
        self.epsilon = self.epsilon
        self.theta = self.theta
        self.tracking = self.tracking
        self.safety_time = self.safety_time
        self.compression = self.compression
        self.local_merges = self.local_merges
        self.debug_plots = self.debug_plots
        self.debug_data = self.debug_data

        if self.probe_path is None:
            # Log error message.
            string = "{}: the probe file must be specified!"
            message = string.format(self.name)
            self.log.error(message)
        else:
            self.probe = load_probe(self.probe_path, radius=self.radius, logger=self.log)
            # Log info message.
            string = "{} reads the probe layout"
            message = string.format(self.name)
            self.log.info(message)

        for directory in [self.debug_plots, self.debug_data]:
            if directory is not None:
                if os.path.exists(directory):
                    shutil.rmtree(directory)
                os.makedirs(directory)

        self.add_input('data', structure='dict')
        self.add_input('pcs', structure='dict')
        self.add_input('peaks', structure='dict')
        self.add_input('mads', structure='dict')
        self.add_output('templates', structure='dict')

        self.thresholds = None

        self._dtype = None
        self._nb_channels = None
        self._nb_samples = None

    def _initialize(self):

        self._spike_width_ = int(self.sampling_rate * self.spike_width * 1e-3)
        self.sign_peaks = []
        self.receive_pcs = True
        self.masks = {}
        if np.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width = (self._spike_width_ - 1) // 2
        self._2_width = 2 * self._width

        if self.safety_time == 'auto':
            self.safety_time = self._spike_width_ // 3
        else:
            self.safety_time = int(self.safety_time * self.sampling_rate * 1e-3)

        if self.alignment:
            num = 5 * self._spike_width_
            self.cdata = np.linspace(-self._width, self._width, num)
            self.xdata = np.arange(-self._2_width, self._2_width + 1)
            self.xoff = len(self.cdata) / 2.

        return

    def _get_all_valid_peaks(self, peaks):

        all_peaks = {}
        for key in peaks.keys():
            all_peaks[key] = set([])
            for channel in peaks[key].keys():
                all_peaks[key] = all_peaks[key].union(peaks[key][channel])

            all_peaks[key] = np.array(list(all_peaks[key]), dtype=np.int32)
            mask = self._is_valid(all_peaks[key])
            all_peaks[key] = all_peaks[key][mask]

            if len(all_peaks[key]) > 0:
                rmin = all_peaks[key].min()
                rmax = all_peaks[key].max()
                diff_times = rmax - rmin
                self.masks[key] = {}
                self.masks[key]['all_times'] = np.zeros((self._nb_channels, diff_times + 1), dtype=np.bool)
                self.masks[key]['min_times'] = np.maximum(all_peaks[key] - rmin - self.safety_time, 0)
                self.masks[key]['max_times'] = np.minimum(all_peaks[key] - rmin + self.safety_time + 1, diff_times)

        return all_peaks

    def _remove_nn_peaks(self, key, peak_idx, channel):

        indices = self.probe.edges[channel]
        min_times_mask = self.masks[key]['min_times'][peak_idx]
        max_times_mask = self.masks[key]['max_times'][peak_idx]
        self.masks[key]['all_times'][indices, min_times_mask:max_times_mask] = True

        return

    def _isolated_peak(self, key, peak_idx, channel):

        indices = self.probe.edges[channel]
        min_times_mask = self.masks[key]['min_times'][peak_idx]
        max_times_mask = self.masks[key]['max_times'][peak_idx]
        myslice = self.masks[key]['all_times'][indices, min_times_mask:max_times_mask]

        return not myslice.any()

    def _is_valid(self, peak):

        if self.alignment:
            cond_1 = (peak >= self._2_width)
            cond_2 = (peak + self._2_width < self._nb_samples)
        else:
            cond_1 = (peak >= self._width)
            cond_2 = (peak + self._width < self._nb_samples)

        return cond_1 & cond_2

    @staticmethod
    def _get_extrema_indices(peak, peaks):

        res = []
        for key in peaks.keys():
            for channel in peaks[key].keys():
                if peak in peaks[key][channel]:
                    res += [int(channel)]

        return res

    def _get_best_channel(self, batch, key, peak, peaks):

        indices = self._get_extrema_indices(peak, peaks)

        if key == 'negative':
            channel = int(np.argmin(batch[peak, indices]))
            is_neg = True
        elif key == 'positive':
            channel = int(np.argmax(batch[peak, indices]))
            is_neg = False
        elif key == 'both':
            v_max = np.max(batch[peak, indices])
            v_min = np.min(batch[peak, indices])
            if np.abs(v_max) > np.abs(v_min):
                channel = int(np.argmax(batch[peak, indices]))
                is_neg = False
            else:
                channel = int(np.argmin(batch[peak, indices]))
                is_neg = True
        else:
            raise NotImplementedError()  # TODO complete.

        return indices[channel], is_neg

    def _get_snippet(self, batch, channel, peak, is_neg):

        indices = self.probe.edges[channel]
        if self.alignment:
            idx = self.chan_positions[channel]
            k_min = peak - self._2_width
            k_max = peak + self._2_width + 1
            zdata = batch[k_min:k_max, indices]
            ydata = np.arange(len(indices))

            if len(ydata) == 1:
                f = scipy.interpolate.UnivariateSpline(self.xdata, zdata, s=0)
                if is_neg:
                    rmin = float(np.argmin(f(self.cdata)) - self.xoff) / 5.0
                else:
                    rmin = float(np.argmax(f(self.cdata)) - self.xoff) / 5.0
                ddata = np.linspace(rmin - self._width, rmin + self._width, self._spike_width_)
                sub_mat = f(ddata).astype(np.float32).reshape(1, self._spike_width_)
            else:
                f = scipy.interpolate.RectBivariateSpline(self.xdata, ydata, zdata, s=0, ky=min(len(ydata) - 1, 3))
                if is_neg:
                    rmin = float(np.argmin(f(self.cdata, idx)[:, 0]) - self.xoff) / 5.0
                else:
                    rmin = float(np.argmax(f(self.cdata, idx)[:, 0]) - self.xoff) / 5.0
                ddata = np.linspace(rmin-self._width, rmin+self._width, self._spike_width_)
                sub_mat = f(ddata, ydata).astype(np.float32)
        else:
            sub_mat = batch[peak - self._width:peak + self._width + 1, indices]

        return sub_mat

    def _configure_input_parameters(self, dtype=None, nb_channels=None, nb_samples=None, **kwargs):

        if dtype is not None:
            self._dtype = dtype
        if nb_channels is not None:
            self._nb_channels = nb_channels
        if nb_samples is not None:
            self._nb_samples = nb_samples

        return

    def _update_initialization(self):

        if self.channels is None:
            self.channels = np.arange(self._nb_channels)

        if self._dtype is not None:
            self.decay_time = self.decay_factor
            self.chan_positions = np.zeros(self._nb_channels, dtype=np.int32)
            for channel in range(self._nb_channels):
                mask = self.probe.edges[channel] == channel
                self.chan_positions[channel] = np.where(mask)[0]

        return

    def _init_data_structures(self):

        self.raw_data = {}
        self.templates = {}
        self.managers = {}
        self.times = {}

        if not np.all(self.pcs[0] == 0):
            self.sign_peaks += ['negative']
        if not np.all(self.pcs[1] == 0):
            self.sign_peaks += ['positive']
        # Log debug message.
        string = "{} will detect peaks {}"
        message = string.format(self.name, self.sign_peaks)
        self.log.debug(message)

        for key in self.sign_peaks:
            self.raw_data[key] = {}
            self.managers[key] = {}
            self.templates[key] = {}
            self.times[key] = {}

            for channel in self.channels:

                params = {
                    'probe': self.probe,
                    'channel': channel,
                    'dispersion': self.dispersion,
                    'mu': self.mu,
                    'epsilon': self.epsilon,
                    'decay': self.decay_time,
                    'theta': self.theta,
                    'n_min': self.n_min,
                    'noise_thr': self.noise_thr,
                    'name': 'OnlineManager for {p} peak on channel {c}'.format(p=key, c=channel),
                    'logger': self.log,
                    'two_components': self.two_components,
                    'debug_plots': self.debug_plots,
                    'local_merges': self.local_merges
                }

                if key == 'negative':
                    params['pca'] = self.pcs[0]
                elif key == 'positive':
                    params['pca'] = self.pcs[1]

                if self.debug_data is not None:
                    self.times[key][channel] = []
                self.templates[key][str(channel)] = {}
                self.managers[key][channel] = OnlineManager(**params)
                self._reset_data_structures(key, channel)

        return

    def _prepare_templates(self, templates, key, channel):

        for ind in templates.keys():
            template = templates[ind]
            template.compress(self.compression)
            template.center(key)
            self.templates[key][str(channel)][str(ind)] = template.to_dict()

        self.to_reset += [(key, channel)]

        return

    def _reset_data_structures(self, key, channel):

        shape = (0, len(self.probe.edges[channel]), self._spike_width_)
        self.raw_data[key][channel] = np.zeros(shape, dtype=np.float32)
        self.templates[key][str(channel)] = {}
        self.times[key][channel] = []

        return

    def _process(self):

        data_packet = self.inputs['data'].receive()
        batch = data_packet['payload']
        if self.is_active:
            peaks_packet = self.inputs['peaks'].receive()
            peaks = peaks_packet['payload']
        else:
            peaks_packet = self.inputs['peaks'].receive(blocking=False)
            peaks = None if peaks_packet is None else peaks_packet['payload']
        mads_packet = self.inputs['mads'].receive(blocking=False)
        self.thresholds = mads_packet['payload'] if mads_packet is not None else self.thresholds

        if self.receive_pcs:
            pcs_packet = self.inputs['pcs'].receive(blocking=False)
            self.pcs = pcs_packet['payload'] if pcs_packet is not None else None

        if self.pcs is not None:  # (i.e. we have already received some principal components).

            if self.receive_pcs:  # (i.e. we need to initialize the block with the principal components).
                # Log info message.
                string = "{} receives the PCA matrices"
                message = string.format(self.name_and_counter)
                self.log.info(message)
                self.receive_pcs = False
                self._init_data_structures()

            if (peaks is not None) and (self.thresholds is not None):  # (i.e. if we receive some peaks and MADs)

                self._measure_time('start', frequency=100)

                self.to_reset = []

                # Synchronize the reception of the peaks with the reception of the data.
                while not self._sync_buffer(peaks, self._nb_samples):
                    peaks_packet = self.inputs['peaks'].receive()
                    peaks = peaks_packet['payload']

                # Set active mode (i.e. use a blocking reception for the peaks).
                if not self.is_active:
                    self._set_active_mode()

                # Retrieve peaks from received buffer.
                offset = peaks.pop('offset')
                all_peaks = self._get_all_valid_peaks(peaks)

                for key in self.sign_peaks:

                    peak_indices = np.random.permutation(np.arange(len(all_peaks[key])))
                    peak_values = np.take(all_peaks[key], peak_indices)

                    for peak_idx, peak in zip(peak_indices, peak_values):

                        channel, is_neg = self._get_best_channel(batch, key, peak, peaks)

                        if self._isolated_peak(key, peak_idx, channel):

                            self._remove_nn_peaks(key, peak_idx, channel)

                            if channel in self.channels:
                                waveforms = self._get_snippet(batch, channel, peak, is_neg).T
                                waveforms = waveforms.reshape(1, waveforms.shape[0], waveforms.shape[1])
                                if is_neg:
                                    key = 'negative'
                                else:
                                    key = 'positive'

                                if not self.managers[key][channel].is_ready:
                                    self.raw_data[key][channel] = np.vstack((self.raw_data[key][channel], waveforms))
                                else:
                                    self.managers[key][channel].update(self.counter, waveforms)
                                if self.debug_data is not None:
                                    self.times[key][channel] += [offset + peak_idx]

                    for channel in self.channels:

                        threshold = self.threshold_factor * self.thresholds[0, channel]
                        self.managers[key][channel].set_physical_threshold(threshold)

                        # Log debug message (if necessary).
                        if self.counter % 50 == 0:
                            string = "{} We have collected {} {} peaks on channel {}"
                            message = string.format(self.name_and_counter, len(self.raw_data[key][channel]), key, channel)
                            self.log.debug(message)

                        if len(self.raw_data[key][channel]) >= \
                                self.nb_waveforms and not self.managers[key][channel].is_ready:
                            # Log debug message.
                            string = "{n} Electrode {k} has obtained {m} {t} waveforms: clustering"
                            message = string.format(n=self.name_and_counter, k=channel, m=self.nb_waveforms, t=key)
                            self.log.debug(message)
                            templates = self.managers[key][channel].initialize(self.counter,
                                                                               self.raw_data[key][channel])
                            self._prepare_templates(templates, key, channel)
                        elif self.managers[key][channel].time_to_cluster(self.nb_waveforms):
                            # Log debug message.
                            string = "{n} Electrode {k} has obtained {m} {t} waveforms: re-clustering"
                            message = string.format(n=self.name_and_counter, k=channel, m=self.nb_waveforms, t=key)
                            self.log.debug(message)
                            templates = self.managers[key][channel].cluster(tracking=self.tracking)
                            self._prepare_templates(templates, key, channel)

                if len(self.to_reset) > 0:
                    self.templates['offset'] = self.counter * self._nb_samples
                    # Prepare output packet.
                    packet = {
                        'number': data_packet['number'],
                        'payload': self.templates,
                    }
                    # # TODO remove the 3 following lines.
                    # string = "{} sends packet: {}"
                    # message = string.format(self.name_and_counter, packet)
                    # self.log.debug(message)
                    # Send templates.
                    self.get_output('templates').send(packet)
                    # Reset data structures.
                    for key, channel in self.to_reset:
                        self._reset_data_structures(key, channel)

                self._measure_time('end', frequency=100)

        return

    def _introspect(self):
        """Introspection of this block for density clustering."""

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self._nb_samples) / self.sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        # Log info message.
        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return
