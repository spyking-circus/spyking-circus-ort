import numpy as np
import os

from circusort.block.block import Block
from circusort.obj.template_store import TemplateStore
from circusort.obj.overlaps_store import OverlapsStore


class Template_fitter(Block):
    """Template fitter

    Attributes:

        init_path: none | string (optional)
            Path to the location used to load templates to initialize the
            dictionary of templates. If equal to None, this dictionary will
            start empty. The default value is None.
    """
    # TODO complete docstring.

    name = "Template fitter"

    params = {
        'templates_init_path': None,
        'overlaps_init_path': None,
        'with_rejected_times': False,
        'sampling_rate': 20e+3,
        'discarding_eoc_from_updater': False,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        # The following lines are useful to avoid some PyCharm's warnings.
        self.templates_init_path = self.templates_init_path
        self.overlaps_init_path = self.overlaps_init_path
        self.with_rejected_times = self.with_rejected_times
        self.sampling_rate = self.sampling_rate
        self.discarding_eoc_from_updater = self.discarding_eoc_from_updater

        self.add_input('updater')
        self.add_input('data')
        self.add_input('peaks')
        self.add_output('spikes', 'dict')

    def _initialize(self):

        self.space_explo = 0.5
        self.nb_chances = 3
        self._overlaps_store = None
        self._template_store = None

        if self.templates_init_path is not None:
            self.templates_init_path = os.path.expanduser(self.templates_init_path)
            self.templates_init_path = os.path.abspath(self.templates_init_path)
            self._initialize_templates()

        # Variables used to handle buffer edges.
        self.x = None  # voltage signal
        self.p = None  # peak time steps
        self.r = {
            'spike_times': np.zeros(0, dtype=np.int32),
            'amplitudes': np.zeros(0, dtype=np.float32),
            'templates': np.zeros(0, dtype=np.int32),
        }  # temporary result

        if self.with_rejected_times:
            self.r.update({
                'rejected_times': np.zeros(0, dtype=np.int32),
                'rejected_amplitudes': np.zeros(0, dtype=np.float32),
            })

        return

    def _initialize_templates(self):

        self._template_store = TemplateStore(self.templates_init_path, mode='r')

        # Log info message.
        string = "{} is initialized with {} templates from {}"
        message = string.format(self.name, self._template_store.nb_templates, self.templates_init_path)
        self.log.info(message)

        self._overlaps_store = OverlapsStore(template_store=self._template_store, path=self.overlaps_init_path, fitting_mode=True)

        # Log info message.
        string = "{} is initialized with precomputed overlaps from {}"
        message = string.format(self.name, self.overlaps_init_path)
        self.log.info(message)

        return

    @property
    def nb_channels(self):
        return self.inputs['data'].shape[1]

    @property
    def nb_samples(self):
        return self.inputs['data'].shape[0]

    @property
    def nb_templates(self):
        if self._overlaps_store is not None:
            return self._overlaps_store.nb_templates
        else:
            return 0

    def _guess_output_endpoints(self):
        if self.templates_init_path is not None:
            self._init_temp_window()

    def _init_temp_window(self):
        self.slice_indices = np.zeros(0, dtype=np.int32)
        self._width = (self._overlaps_store.temporal_width - 1) // 2
        self._2_width = 2 * self._width
        temp_window = np.arange(-self._width, self._width + 1)
        buffer_size = 2 * self.nb_samples
        for idx in range(self.nb_channels):
            self.slice_indices = np.concatenate((self.slice_indices, idx * buffer_size + temp_window))

    def _reset_result(self):

        self.result = {
            'spike_times': self.r['spike_times'],
            'amplitudes': self.r['amplitudes'],
            'templates': self.r['templates'],
            'offset': self.offset,
        }
        if self.with_rejected_times:
            self.result.update({
                'rejected_times': self.r['rejected_times'],
                'rejected_amplitudes': self.r['rejected_amplitudes']
            })
        self.r = {
            'spike_times': np.zeros(0, dtype=np.int32),
            'amplitudes': np.zeros(0, dtype=np.float32),
            'templates': np.zeros(0, dtype=np.int32),
        }
        if self.with_rejected_times:
            self.r.update({
                'rejected_times': np.zeros(0, dtype=np.int32),
                'rejected_amplitudes': np.zeros(0, dtype=np.float32)
            })

        return

    def _extract_waveforms(self, peak_time_steps):
        """Extract waveforms from buffer

        Attributes:
            peak_time_steps: np.array
                Peak time steps. Array of shape (number of peaks,).

        Return:
            waveforms: np.array
                Extracted waveforms. Array of shape (waveform size, number of peaks), where waveform size is equal to
                number of channels x number of samples.
        """

        batch = self.x.T.flatten()
        nb_peaks = len(peak_time_steps)
        waveforms = np.zeros((self._overlaps_store.nb_elements, nb_peaks), dtype=np.float32)
        for k, peak_time_step in enumerate(peak_time_steps):
            waveforms[:, k] = batch[self.slice_indices + peak_time_step]

        return waveforms

    def _fit_chunk(self):

        # # Log some information.
        # string = "{} fits spikes... ({} templates)"
        # message = string.format(self.name_and_counter, self.nb_templates)
        # self.log.debug(message)

        # Reset result.
        self._reset_result()

        # Compute the number of peaks in the current chunk.
        p_min = 1 * self.nb_samples - self._width  # start index of the work area
        p_max = 2 * self.nb_samples - self._width  # end index of the work area
        is_in_work_area = np.logical_and(p_min <= self.p, self.p < p_max)
        nb_peaks = np.count_nonzero(is_in_work_area)
        peaks = self.p[is_in_work_area]

        # If there is at least one peak in the work area...
        if 0 < nb_peaks:

            # Extract waveforms from buffer.
            waveforms = self._extract_waveforms(peaks)

            # Compute the scalar products between waveforms and templates.
            scalar_products = self._overlaps_store.dot(waveforms)

            # Initialize the failure counter of each peak.
            nb_failures = np.zeros(nb_peaks, dtype=np.int32)
            # Initialize the matching matrix.
            mask = np.ones((self._overlaps_store.nb_templates, nb_peaks), dtype=np.int32)
            # Filter scalar products of the first component of each template.
            sub_b = scalar_products[:self._overlaps_store.nb_templates, :]

            # TODO rewrite condition according to the 3 last lines of the nested while loop.
            # while not np.all(nb_failures == self.max_nb_trials):
            while np.mean(nb_failures) < self.nb_chances:

                # Set scalar products of tested matchings to zero.
                data = sub_b * mask
                # Sort peaks by decreasing highest scalar product with all the templates.
                peak_indices = np.argsort(np.max(data, axis=0))[::-1]
                # TODO remove peaks with scalar products equal to zero?
                # TODO consider the absolute values of the scalar products?

                for peak_index in peak_indices:

                    # Find the best template.
                    peak_scalar_products = np.take(sub_b, peak_index, axis=1)
                    best_template_index = np.argmax(peak_scalar_products, axis=0)

                    # Compute the best amplitude.
                    best_amplitude = sub_b[best_template_index, peak_index] / self._overlaps_store.nb_elements
                    if self._overlaps_store.two_components:
                        best_scalar_product = scalar_products[best_template_index + self.nb_templates, peak_index]
                        best_amplitude_2 = best_scalar_product / self._overlaps_store.nb_elements

                    # Compute the best normalized amplitude.
                    best_amplitude_ = best_amplitude / self._overlaps_store.norms['1'][best_template_index]
                    if self._overlaps_store.two_components:
                        best_amplitude_2_ = best_amplitude_2 / self._overlaps_store.norms['2'][best_template_index]
                        _ = best_amplitude_2_  # TODO complete.

                    # Verify amplitude constraint.
                    a_min = self._overlaps_store.amplitudes[best_template_index, 0]
                    a_max = self._overlaps_store.amplitudes[best_template_index, 1]

                    if (a_min <= best_amplitude_) & (best_amplitude_ <= a_max):
                        # Keep the matching.
                        peak_time_step = peaks[peak_index]
                        # # Compute the neighboring peaks.
                        # # TODO use this definition of `is_neighbor` instead of the other.
                        # is_neighbor = np.abs(peaks - peak_index) <= 2 * self._width

                        # Update scalar products.
                        # TODO simplify the following 11 lines.
                        tmp = np.dot(np.ones((1, 1), dtype=np.int32), np.reshape(peaks, (1, nb_peaks)))
                        tmp -= np.array([[peak_time_step]])
                        is_neighbor = np.abs(tmp) <= self._2_width
                        ytmp = tmp[0, is_neighbor[0, :]] + self._2_width
                        indices = np.zeros((self._overlaps_store.size, len(ytmp)), dtype=np.int32)
                        indices[ytmp, np.arange(len(ytmp))] = 1

                        tmp1_ = self._overlaps_store.get_overlaps(best_template_index, '1')
                        tmp1 = tmp1_.multiply(-best_amplitude).dot(indices)
                        scalar_products[:, is_neighbor[0, :]] += tmp1

                        if self._overlaps_store.two_components:
                            tmp2_ = self._overlaps_store.get_overlaps(best_template_index, '2')
                            tmp2 = tmp2_.multiply(-best_amplitude_2).dot(indices)
                            scalar_products[:, is_neighbor[0, :]] += tmp2

                        # Add matching to the result.
                        self.r['spike_times'] = np.concatenate((self.r['spike_times'], [peak_time_step]))
                        self.r['amplitudes'] = np.concatenate((self.r['amplitudes'], [best_amplitude_]))
                        self.r['templates'] = np.concatenate((self.r['templates'], [best_template_index]))
                        # Mark current matching as tried.
                        mask[best_template_index, peak_index] = 0
                    else:
                        # Reject the matching.
                        # Update failure counter of the peak.
                        nb_failures[peak_index] += 1
                        # If the maximal number of failures is reached then mark peak as solved (i.e. not fitted).
                        if nb_failures[peak_index] == self.nb_chances:
                            mask[:, peak_index] = 0
                        else:
                            mask[best_template_index, peak_index] = 0
                        # Add reject to the result if necessary.
                        if self.with_rejected_times:
                            self.r['rejected_times'] = np.concatenate((self.r['rejected_times'],
                                                                       [peaks[peak_index]]))
                            self.r['rejected_amplitudes'] = np.concatenate((self.r['rejected_amplitudes'],
                                                                            [best_amplitude_]))

            # Handle result.
            is_in_result = self.r['spike_times'] < self.nb_samples
            self.result['spike_times'] = np.concatenate((self.result['spike_times'],
                                                         self.r['spike_times'][is_in_result]))
            self.result['amplitudes'] = np.concatenate((self.result['amplitudes'],
                                                        self.r['amplitudes'][is_in_result]))
            self.result['templates'] = np.concatenate((self.result['templates'],
                                                       self.r['templates'][is_in_result]))
            self.r['spike_times'] = self.r['spike_times'][~is_in_result] - self.nb_samples
            self.r['amplitudes'] = self.r['amplitudes'][~is_in_result]
            self.r['templates'] = self.r['templates'][~is_in_result]
            if self.with_rejected_times:
                is_in_result = self.r['rejected_times'] < self.nb_samples
                self.result['rejected_times'] = np.concatenate((self.result['rejected_times'],
                                                                self.r['rejected_times'][is_in_result]))
                self.result['rejected_amplitudes'] = np.concatenate((self.result['rejected_amplitudes'],
                                                                     self.r['rejected_amplitudes'][is_in_result]))
                self.r['rejected_times'] = self.r['rejected_times'][~is_in_result] - self.nb_samples
                self.r['rejected_amplitudes'] = self.r['rejected_amplitudes'][~is_in_result]
            indices = np.argsort(self.result['spike_times'])
            for key in ['spike_times', 'amplitudes', 'templates']:
                self.result[key] = self.result[key][indices]
            if self.with_rejected_times:
                indices = np.argsort(self.result['rejected_times'])
                for key in ['rejected_times', 'rejected_amplitudes']:
                    self.result[key] = self.result[key][indices]

            # Log fitting result.
            nb_spike_times = len(self.result['spike_times'])
            if nb_spike_times > 0:
                string = "{} fitted {} spikes ({} templates)"
                message = string.format(self.name_and_counter, nb_spike_times, self.nb_templates)
                self.log.debug(message)
            else:
                string = "{} fitted no spikes ({} templates)"
                message = string.format(self.name_and_counter, self.nb_templates)
                self.log.debug(message)

        else:  # nb_peaks == 0

            string = "{} can't fit spikes ({} templates)"
            message = string.format(self.name_and_counter, self.nb_templates)
            self.log.debug(message)

        return

    @staticmethod
    def _merge_peaks(peaks):
        """Merge positive and negative peaks from all the channels"""

        time_steps = set([])
        keys = [key for key in peaks.keys() if key not in ['offset']]
        for key in keys:
            for channel in peaks[key].keys():
                time_steps = time_steps.union(peaks[key][channel])
        time_steps = np.array(list(time_steps), dtype=np.int32)
        time_steps = np.sort(time_steps)

        return time_steps

    def _process(self):

        # Update internal variables to handle buffer edges.
        if self.counter == 0:
            shape = (2 * self.nb_samples, self.nb_channels)
            self.x = np.zeros(shape, dtype=np.float32)
            self.x[self.nb_samples:, :] = self.inputs['data'].receive()
        else:
            self.x[:self.nb_samples, :] = self.x[self.nb_samples:, :]
            self.x[self.nb_samples:, :] = self.inputs['data'].receive()

        if self.is_active:
            peaks = self.inputs['peaks'].receive()
        else:
            peaks = self.inputs['peaks'].receive(blocking=False)

        updater = self.inputs['updater'].receive(blocking=False, discarding_eoc=self.discarding_eoc_from_updater)

        if updater is not None:

            self._measure_time('update_start', frequency=1)

            indices = updater.get('indices', None)
            # Create the template dictionary if necessary.
            if self._template_store is None:
                self._template_store = TemplateStore(updater['template_store'], mode='r')
                self._overlaps_store = OverlapsStore(template_store=self._template_store,
                                                     path=updater['overlaps']['path'], fitting_mode=True)
                self._init_temp_window()
            else:
                laziness = updater['overlaps']['path'] is None
                self._overlaps_store.update(indices, laziness=laziness)

            self._measure_time('update_end', frequency=1)

        if peaks is not None:

            if self.nb_templates > 0:
                self._measure_time('start', frequency=100)

            self.offset = (self.counter - 1) * self.nb_samples

            if not self.is_active:
                # Handle peaks.
                p = self.nb_samples + self._merge_peaks(peaks)
                self.p = p
                # Synchronize peak reception.
                while not self._sync_buffer(peaks, self.nb_samples):
                    peaks = self.inputs['peaks'].receive()
                    p = self.nb_samples + self._merge_peaks(peaks)
                    self.p = self.p - self.nb_samples
                    self.p = self.p[0 <= self.p]
                    self.p = np.concatenate((self.p, p))
                # Set active mode.
                self._set_active_mode()
            else:

                # Handle peaks.
                p = self.nb_samples + self._merge_peaks(peaks)
                self.p = self.p - self.nb_samples
                self.p = self.p[0 <= self.p]
                self.p = np.concatenate((self.p, p))

            if (peaks['offset'] - self.offset) != self.nb_samples:
                self.log.info('Fitter and peaks not in sync!')

            elif self.nb_templates > 0:

                self._fit_chunk()
                if 0 < self.counter:
                    self.output.send(self.result)

                self._measure_time('end', frequency=100)

        return

    def _introspect(self):
        # TODO add docstring.

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self.nb_samples) / self.sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return
