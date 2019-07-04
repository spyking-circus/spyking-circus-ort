
# coding: utf8
from .block import Block
import numpy as np
import os
import shutil

from circusort.io.probe import load_probe
from circusort.io.snippets import empty as empty_snippets
from circusort.utils.clustering import OnlineManager
from circusort.obj.buffer import Buffer


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
        'threshold_factor': 6.0,
        'alignment': True,
        'sampling_rate': 20.e+3,  # Hz
        'spike_width': 3.0,  # ms
        'spike_jitter': 0.1,  # ms
        'spike_sigma': 0.0,  # µV
        'nb_waveforms': 1000,
        'nb_waveforms_tracking': 1000,
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
        'debug_ground_truth_templates': None,
        'debug_file_format': 'png',
        'debug_data': None,
        'smart_select': 'ransac',
        'hanning_filtering' : False
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        # The following lines are useful to avoid some PyCharm's warnings.
        self.threshold_factor = self.threshold_factor
        self.alignment = self.alignment
        self.sampling_rate = self.sampling_rate
        self.spike_width = self.spike_width
        self.spike_jitter = self.spike_jitter
        self.spike_sigma = self.spike_sigma
        self.nb_waveforms = self.nb_waveforms
        self.nb_waveforms_tracking = self.nb_waveforms_tracking
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
        self.debug_ground_truth_templates = self.debug_ground_truth_templates
        self.debug_file_format = self.debug_file_format
        self.debug_data = self.debug_data
        self.smart_select = self.smart_select
        self.hanning_filtering = self.hanning_filtering

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
                # Log info message.
                string = "{} creates directory {}"
                message = string.format(self.name, directory)
                self.log.info(message)

        self.add_input('data', structure='dict')
        self.add_input('pcs', structure='dict')
        self.add_input('peaks', structure='dict')
        self.add_input('mads', structure='dict')
        self.add_output('templates', structure='dict')

        self.thresholds = None

        self._dtype = None
        self._nb_channels = None
        self._nb_samples = None

        self.inodes = np.zeros(self.probe.total_nb_channels, dtype=np.int32)
        self.inodes[self.probe.nodes] = np.argsort(self.probe.nodes)

    def _initialize(self):

        self.batch = Buffer(self.sampling_rate, self.spike_width, self.spike_jitter,
                            alignment=self.alignment, probe=self.probe)
        self.sign_peaks = []
        self.receive_pcs = True
        self.masks = {}
        self.safety_time = self.batch.get_safety_time(self.safety_time)

        return

    def _get_all_valid_peaks(self, peaks):

        all_peaks = {}
        for key in peaks.keys():
            all_peaks[key] = set([])
            for channel in peaks[key].keys():
                all_peaks[key] = all_peaks[key].union(peaks[key][channel])

            all_peaks[key] = np.array(list(all_peaks[key]), dtype=np.int32)
            mask = self.batch.valid_peaks(all_peaks[key])
            all_peaks[key] = all_peaks[key][mask]

            if len(all_peaks[key]) > 0:
                r_min = all_peaks[key].min()
                r_max = all_peaks[key].max()
                diff_times = r_max - r_min
                self.masks[key] = {}
                self.masks[key]['all_times'] = np.zeros((self._nb_channels, diff_times + 1), dtype=np.bool)
                self.masks[key]['min_times'] = np.maximum(all_peaks[key] - r_min - self.safety_time, 0)
                self.masks[key]['max_times'] = np.minimum(all_peaks[key] - r_min + self.safety_time + 1, diff_times)

        return all_peaks

    def _remove_nn_peaks(self, key, peak_idx, channel):

        indices = self.inodes[self.probe.edges[self.probe.nodes[channel]]]
        min_times_mask = self.masks[key]['min_times'][peak_idx]
        max_times_mask = self.masks[key]['max_times'][peak_idx]
        self.masks[key]['all_times'][indices, min_times_mask:max_times_mask] = True

        return

    def _isolated_peak(self, key, peak_idx, channel):

        indices = self.inodes[self.probe.edges[self.probe.nodes[channel]]]
        min_times_mask = self.masks[key]['min_times'][peak_idx]
        max_times_mask = self.masks[key]['max_times'][peak_idx]
        my_slice = self.masks[key]['all_times'][indices, min_times_mask:max_times_mask]

        return not my_slice.any()

    @staticmethod
    def _get_extrema_indices(peak, peaks):

        res = []
        for key in peaks.keys():
            for channel in peaks[key].keys():
                if peak in peaks[key][channel]:
                    res += [int(channel)]

        return res

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
            self.channels = self.inodes[self.probe.nodes]

        if self._dtype is not None:
            self.decay_time = self.decay_factor
            # self.chan_positions = {}
            # for channel in self.channels:
            #     mask = self.probe.edges[channel] == channel
            #     self.chan_positions[channel] = np.where(mask)[0]

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
                    'decay': self.decay_time,
                    'mu': self.mu,
                    'epsilon': self.epsilon,
                    'theta': self.theta,
                    'dispersion': self.dispersion,
                    'n_min': self.n_min,
                    'noise_thr': self.noise_thr,
                    'pca': None,  # see below
                    'logger': self.log,
                    'two_components': self.two_components,
                    'name': 'OnlineManager for {p} peak on channel {c}'.format(p=key, c=channel),
                    'debug_plots': self.debug_plots,
                    'debug_ground_truth_templates': self.debug_ground_truth_templates,
                    'local_merges': self.local_merges,
                    'debug_file_format': self.debug_file_format,
                    'sampling_rate': self.sampling_rate,
                    'smart_select': self.smart_select,
                    'hanning_filtering': self.hanning_filtering
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

        self.raw_data[key][channel] = empty_snippets()
        self.templates[key][str(channel)] = {}
        self.times[key][channel] = []

        return

    def _process(self):

        data_packet = self.inputs['data'].receive()
        data = data_packet['payload']
        offset = data_packet['number'] * self._nb_samples
        self.batch.update(data, offset=offset)
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
                if self.debug_data is not None:
                    path = os.path.join(self.debug_data, 'pca.npy')
                    np.save(path, self.pcs)

            if (peaks is not None) and (self.thresholds is not None):  # (i.e. if we receive some peaks and MADs)

                self._measure_time('start')

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

                        channels = self._get_extrema_indices(peak, peaks)
                        best_channel, peak_type = self.batch.get_best_channel(channels, peak, key)

                        if self._isolated_peak(peak_type, peak_idx, best_channel):

                            self._remove_nn_peaks(peak_type, peak_idx, best_channel)
                            
                            if best_channel in self.channels:
                                channels = self.inodes[self.probe.edges[self.probe.nodes[best_channel]]]
                                waveforms = self.batch.get_snippet(channels, peak, peak_type=peak_type,
                                                                   ref_channel=best_channel)

                                online_manager = self.managers[key][best_channel]
                                if not online_manager.is_ready:
                                    self.raw_data[key][best_channel].add(waveforms)
                                else:
                                    online_manager.update(self.counter, waveforms)

                                if self.debug_data is not None:
                                    self.times[key][best_channel] += [offset + peak_idx]

                    for channel in self.channels:

                        online_manager = self.managers[key][channel]

                        threshold = self.threshold_factor * self.thresholds[0, channel]
                        online_manager.set_physical_threshold(threshold)

                        # Log debug message (if necessary).
                        if self.counter % 50 == 0:
                            nb_peaks = len(self.raw_data[key][channel])
                            string = "{} We have collected {} {} peaks on channel {}"
                            message = string.format(self.name_and_counter, nb_peaks, key, channel)
                            self.log.debug(message)

                        if len(self.raw_data[key][channel]) >= self.nb_waveforms and not online_manager.is_ready:
                            # Log debug message.
                            string = "{} Electrode {} has obtained {} {} waveforms: clustering"
                            message = string.format(self.name_and_counter, channel, self.nb_waveforms, key)
                            self.log.debug(message)
                            # First clustering.
                            templates = online_manager.initialize(self.counter, self.raw_data[key][channel])
                            self._prepare_templates(templates, key, channel)
                        elif self.managers[key][channel].time_to_cluster(nb_updates=self.nb_waveforms_tracking):
                            # Log debug message.
                            string = "{} Electrode {} has obtained {} {} waveforms: re-clustering"
                            message = string.format(self.name_and_counter, channel, self.nb_waveforms_tracking, key)
                            self.log.debug(message)
                            # Re-clustering.
                            templates = online_manager.cluster(tracking=self.tracking)
                            self._prepare_templates(templates, key, channel)

                if len(self.to_reset) > 0:
                    self.templates['offset'] = self.counter * self._nb_samples
                    # Prepare output packet.
                    packet = {
                        'number': data_packet['number'],
                        'payload': self.templates,
                    }
                    # Send templates.
                    self.get_output('templates').send(packet)
                    # Log debug message.
                    string = "{} sends output packet"
                    message = string.format(self.name_and_counter)
                    self.log.debug(message)
                    # Reset data structures.
                    for key, channel in self.to_reset:
                        self._reset_data_structures(key, channel)

                self._measure_time('end')

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
