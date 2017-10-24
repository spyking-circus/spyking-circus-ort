from .block import Block

import numpy as np
import os
from scipy.sparse import csr_matrix, vstack

from circusort.io.template import TemplateStore
# from circusort.io.template import OverlapStore
# from circusort.io.utils import load_pickle


class Template_fitter(Block):
    """Template fitter

    Attributes:
        spike_width: float (optional)
            Spike width in time [ms]. the default value is 5.0.
        sampling_rate: float (optional)
            Sampling rate [Hz]. The default value is 20e+3.
        two_components: boolean (optional)
            The default value is False.
        init_path: none | string (optional)
            Path to the location used to load templates to initialize the
            dictionary of templates. If equal to None, this dictionary will
            start empty. The default value is None.
    """
    # TODO complete docstring.

    name = "Template fitter"

    params = {
        'spike_width': 5.0,
        'sampling_rate': 20000,
        'two_components': False,
        'init_path': None,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_input('updater')
        self.add_input('data')
        self.add_input('peaks')
        self.add_output('spikes', 'dict')

    def _initialize(self):

        self.space_explo   = 0.5
        self.nb_chances    = 3
        self._spike_width_ = int(self.sampling_rate * self.spike_width * 1e-3)
        self.template_store = None
        self.norms = np.zeros(0, dtype=np.float32)
        self.amplitudes = np.zeros((0, 2), dtype=np.float32)
        self.templates = None
        self.variables = ['norms', 'templates', 'amplitudes']
        if self.two_components:
            self.norms2 = np.zeros(0, dtype=np.float32)
            self.variables += ['norms2', 'templates2']

        if np.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width = (self._spike_width_ - 1) // 2
        self._overlap_size = 2 * self._spike_width_ - 1

        if self.init_path is not None:
            self.init_path = os.path.expanduser(self.init_path)
            self.init_path = os.path.abspath(self.init_path)
            self._initialize_templates()

        return

    def _initialize_templates(self):

        assert self.template_store is None

        self.template_store = TemplateStore(self.init_path,
                                            initialized=True,
                                            mode='r',
                                            two_components=self.two_components,
                                            N_t=self._spike_width_)
        nb_templates = self.template_store.nb_templates
        indices = [i for i in range(nb_templates)]
        data = self.template_store.get(indices=indices,
                                       variables=self.variables)
        self.norms = np.concatenate((self.norms, data.pop('norms')))
        self.amplitudes = np.vstack((self.amplitudes, data.pop('amplitudes')))
        self.templates = vstack((data.pop('templates').T,), 'csr')
        if self.two_components:
            self.norms2 = np.concatenate((self.norms2, data.pop('norms2')))
            self.templates = vstack((self.templates, data.pop('templates2').T), 'csr')
        self.overlaps = {}

        info_msg = "{} is initialized with {} templates from {}"
        self.log.info(info_msg.format(self.name, nb_templates, self.init_path))

        return

    @property
    def nb_channels(self):

        return self.inputs['data'].shape[1]

    @property
    def nb_samples(self):

        return self.inputs['data'].shape[0]

    @property
    def nb_templates(self):

        if self.two_components:
            return self.templates.shape[0] / 2
        else:
            return self.templates.shape[0]

    def _guess_output_endpoints(self):

        self._nb_elements = self.nb_channels * self._spike_width_
        if self.templates is None:
            self.templates = csr_matrix((0, self._nb_elements), dtype=np.float32)
        self.slice_indices = np.zeros(0, dtype=np.int32)
        self.all_cols = np.arange(self.nb_channels * self._spike_width_)
        self.all_delays = np.arange(1, self._spike_width_ + 1)

        temp_window = np.arange(-self._width, self._width + 1)
        for idx in range(self.nb_channels):
            self.slice_indices = np.concatenate((self.slice_indices, self.nb_samples * idx + temp_window))

        return

    def _is_valid(self, peak_step):

        i_min = self._width
        i_max = self.nb_samples - self._width
        is_valid = (i_min <= peak_step) & (peak_step < i_max)

        return is_valid

    def _get_all_valid_peaks(self, peak_steps):

        all_peak_steps = set([])
        for key in peak_steps.keys():
            for channel in peak_steps[key].keys():
                all_peak_steps = all_peak_steps.union(peak_steps[key][channel])
        all_peak_steps = np.array(list(all_peak_steps), dtype=np.int32)

        mask = self._is_valid(all_peak_steps)
        all_valid_peak_steps = all_peak_steps[mask]

        return all_valid_peak_steps

    def _reset_result(self):

        self.result = {
            'spike_times': np.zeros(0, dtype=np.int32),
            'amplitudes': np.zeros(0, dtype=np.float32),
            'templates': np.zeros(0, dtype=np.int32),
            'offset': self.offset,
        }

        return

    def _update_overlaps(self, sources):

        sources = np.array(sources, dtype=np.int32)

        if self.two_components:
            sources = np.concatenate((sources, sources + self.nb_templates))

        selection = list(set(sources).difference(self.overlaps.keys()))

        if len(selection) > 0:

            tmp_loc_c1 = self.templates[selection]
            tmp_loc_c2 = self.templates

            all_x = np.zeros(0, dtype=np.int32)
            all_y = np.zeros(0, dtype=np.int32)
            all_data = np.zeros(0, dtype=np.float32)

            for idelay in self.all_delays:
                scols = np.where(self.all_cols % self._spike_width_ < idelay)[0]
                tmp_1 = tmp_loc_c1[:, scols]
                scols = np.where(self.all_cols % self._spike_width_ >= (self._spike_width_ - idelay))[0]
                tmp_2 = tmp_loc_c2[:, scols]
                data = tmp_1.dot(tmp_2.T).toarray()

                dx, dy = data.nonzero()
                data = data[data.nonzero()].ravel()

                all_x = np.concatenate((all_x, dx * self.templates.shape[0] + dy))
                all_y = np.concatenate((all_y, (idelay - 1) * np.ones(len(dx), dtype=np.int32)))
                all_data = np.concatenate((all_data, data))

                if idelay < self._spike_width_:
                    all_x = np.concatenate((all_x, dy * len(selection) + dx))
                    all_y = np.concatenate((all_y, (2 * self._spike_width_ - idelay - 1) * np.ones(len(dx), dtype=np.int32)))
                    all_data = np.concatenate((all_data, data))

            overlaps = csr_matrix((all_data, (all_x, all_y)), shape=(self.templates.shape[0] * len(selection), self._overlap_size))

            for count, c in enumerate(selection):
                self.overlaps[c] = overlaps[count * self.templates.shape[0]:(count + 1) * self.templates.shape[0]]

        return

    def _extract_waveforms(self, batch, peak_time_steps):
        """Extract waveforms from buffer

        Attributes:
            batch: np.array
                Buffer of data. Array of shape (number of channels, number of samples).
            peak_time_steps: np.array
                Peak time steps. Array of shape (number of peaks,).

        Return:
            waveforms: np.array
                Extracted waveforms. Array of shape (waveform size, number of peaks), where waveform size is equal to
                number of channels x number of samples.
        """

        batch = batch.T.flatten()
        waveform_size = self.nb_channels * self._spike_width_
        nb_peaks = len(peak_time_steps)
        waveforms = np.zeros((waveform_size, nb_peaks), dtype=np.float32)
        for k, peak_time_step in enumerate(peak_time_steps):
            waveforms[:, k] = batch[self.slice_indices + peak_time_step]

        return waveforms

    def _fit_chunk(self, batch, peaks):

        # Reset result.
        self._reset_result()

        # Filter valid detected peaks (i.e. peaks outside buffer edges).
        peaks = self._get_all_valid_peaks(peaks)
        nb_peaks = len(peaks)

        # If there is at least one peak...
        if nb_peaks > 0:

            # Extract waveforms from buffer.
            waveforms = self._extract_waveforms(batch, peaks)
            # Compute the scalar products between waveforms and templates.
            scalar_products = self.templates.dot(waveforms)
            # Initialize the failure counter of each peak.
            nb_failures = np.zeros(nb_peaks, dtype=np.int32)
            # Initialize the matching matrix.
            mask = np.ones((self.nb_templates, nb_peaks), dtype=np.int32)
            # Filter scalar products of the first component of each template.
            sub_b = scalar_products[:self.nb_templates, :]

            # TODO rewrite condition according to the 3 last lines of the nested while loop.
            # while not np.all(nb_failures == self.max_nb_trials):
            while np.mean(nb_failures) < self.nb_chances:

                # Set scalar products of tested matchings to zero.
                data = sub_b * mask
                # Sort peaks by decreasing highest scalar product with all the templates.
                peak_indices = np.argsort(np.max(data, axis=0))[::-1]
                # TODO remove peaks with scalar products equal to zero?

                for peak_index in peak_indices:

                    # Find the best template.
                    peak_scalar_products = np.take(sub_b, peak_index, axis=1)
                    best_template_index = np.argmax(peak_scalar_products, axis=0)

                    # Compute the best amplitude.
                    best_amplitude = sub_b[best_template_index, peak_index] / self._nb_elements
                    if self.two_components:
                        best_amplitude_2 = scalar_products[best_template_index + self.nb_templates, peak_index] / self._nb_elements
                    # Compute the best normalized amplitude.
                    best_amplitude_ = best_amplitude / np.take(self.norms, best_template_index)
                    if self.two_components:
                        best_amplitude_2_ = best_amplitude_2 / np.take(self.norms2, best_template_index)

                    # Verify amplitude constraint.
                    a_min = self.amplitudes[best_template_index, 0]
                    a_max = self.amplitudes[best_template_index, 1]
                    if (a_min <= best_amplitude_) & (best_amplitude_ <= a_max):
                        # Keep the matching.
                        peak_time_step = peaks[peak_index]
                        # Compute the neighboring peaks.
                        is_neighbor = np.abs(peaks - peak_index) <= 2 * self._width
                        # Update the overlapping matrix.
                        self._update_overlaps(np.array([best_template_index]))
                        # Update scalar products.
                        # TODO simplify the following 11 lines.
                        tmp = np.dot(np.ones((1, 1), dtype=np.int32), np.reshape(peaks, (1, nb_peaks)))
                        tmp -= np.array([[peak_time_step]])
                        is_neighbor = np.abs(tmp) <= 2 * self._width
                        ytmp = tmp[0, is_neighbor[0, :]] + 2 * self._width
                        indices = np.zeros((self._overlap_size, len(ytmp)), dtype=np.int32)
                        indices[ytmp, np.arange(len(ytmp))] = 1
                        tmp1 = self.overlaps[best_template_index].multiply(-best_amplitude).dot(indices)
                        scalar_products[:, is_neighbor[0, :]] += tmp1
                        if self.two_components:
                            tmp2 = self.overlaps[best_template_index + self.nb_templates].multiply(-best_amplitude_2).dot(indices)
                            scalar_products[:, is_neighbor[0, :]] += tmp2
                        # Verify if peak do not stand in the buffer edges.
                        # TODO change if we have a proper handling of the borders.
                        # min_time_step = 2 * self._width
                        # max_time_step = self.nb_samples - 2 * self._width
                        min_time_step = self._width
                        max_time_step = self.nb_samples - self._width
                        if (min_time_step <= peak_time_step) & (peak_time_step < max_time_step):
                            # Keep the matching.
                            self.result['spike_times'] = np.concatenate((self.result['spike_times'], [peak_time_step]))
                            self.result['amplitudes'] = np.concatenate((self.result['amplitudes'], [best_amplitude_]))
                            self.result['templates'] = np.concatenate((self.result['templates'], [best_template_index]))
                        else:
                            # Throw away the matching.
                            pass
                        # Mark current matching as tried.
                        mask[best_template_index, peak_index] = 0
                    else:
                        # Reject the matching.
                        # Update failure counter of the peak.
                        nb_failures[peak_index] += 1
                        # If the maximal number of failures is reached then mark as solved (i.e. not fitted).
                        if nb_failures[peak_index] == self.nb_chances:
                            mask[:, peak_index] = 0
                        else:
                            mask[best_template_index, peak_index] = 0

            # Log fitting result.
            if len(self.result['spike_times']) > 0:
                self.log.debug('{n} fitted {k} spikes from {m} templates'.format(n=self.name_and_counter, k=len(self.result['spike_times']), m=self.nb_templates))
            else:
                self.log.debug('{n} fitted no spikes from {s} peaks'.format(n=self.name_and_counter, s=nb_peaks))

        return

    def _process(self):

        batch = self.inputs['data'].receive()
        if self.is_active:
            peaks = self.inputs['peaks'].receive()
        else:
            peaks = self.inputs['peaks'].receive(blocking=False)
        updater = self.inputs['updater'].receive(blocking=False)

        if updater is not None:

            if self.template_store is None:
                self.template_store = TemplateStore(updater['templates_file'], 'r', self.two_components)

            data = self.template_store.get(updater['indices'], variables=self.variables)

            self.norms = np.concatenate((self.norms, data.pop('norms')))
            self.amplitudes = np.vstack((self.amplitudes, data.pop('amplitudes')))
            self.templates = vstack((self.templates, data.pop('templates').T), 'csr')

            if self.two_components:
                self.norms2 = np.concatenate((self.norms2, data.pop('norms2')))
                self.templates = vstack((self.templates, data.pop('templates2').T), 'csr')

            self.overlaps = {}

        if peaks is not None:

            if not self.is_active:
                # Synchronize peak reception.
                while not self._sync_buffer(peaks, self.nb_samples):
                    peaks = self.inputs['peaks'].receive()
                # Set active mode.
                self._set_active_mode()

            _ = peaks.pop('offset')
            self.offset = self.counter * self.nb_samples

            if self.nb_templates > 0:

                self._fit_chunk(batch, peaks)
                self.output.send(self.result)

        return
