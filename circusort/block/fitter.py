import numpy as np
import os

from circusort.block.block import Block
from circusort.obj.template_store import TemplateStore
from circusort.obj.overlaps_dictionary import OverlapsDictionary


class Fitter(Block):
    """Fitter

    Attributes:

        init_path: none | string (optional)
            Path to the location used to load templates to initialize the
            dictionary of templates. If equal to None, this dictionary will
            start empty. The default value is None.
    """
    # TODO complete docstring.

    name = "Fitter"

    params = {
        'init_path': None,
        'with_rejected_times': False,
        'sampling_rate': 20e+3,
        'discarding_eoc_from_updater': False,
        '_nb_fitters': 1,
        '_fitter_id': 0,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        # The following lines are useful to avoid some PyCharm's warnings.
        self.init_path = self.init_path
        self.with_rejected_times = self.with_rejected_times
        self.sampling_rate = self.sampling_rate
        self.discarding_eoc_from_updater = self.discarding_eoc_from_updater
        self._nb_fitters = self._nb_fitters
        self._fitter_id = self._fitter_id

        self.add_input('updater')
        self.add_input('data')
        self.add_input('peaks')
        self.add_output('spikes', 'dict')

    def _initialize(self):

        self.space_explo = 0.5
        self.nb_chances = 3
        self.overlaps_store = None

        if self.init_path is not None:
            self.init_path = os.path.expanduser(self.init_path)
            self.init_path = os.path.abspath(self.init_path)
            self._initialize_templates()

        # Variables used to handle buffer edges.
        self.x = None  # voltage signal
        self.p = None  # peak time steps
        self.r = {  # temporary result
            'spike_times': np.zeros(0, dtype=np.int32),
            'amplitudes': np.zeros(0, dtype=np.float32),
            'templates': np.zeros(0, dtype=np.int32),
        }
        if self.with_rejected_times:
            self.r.update({
                'rejected_times': np.zeros(0, dtype=np.int32),
                'rejected_amplitudes': np.zeros(0, dtype=np.float32),
            })

        return

    def _initialize_templates(self):

        self.template_store = TemplateStore(self.init_path, mode='r')
        self.overlaps_store = OverlapsDictionary(self.template_store)

        string = "{} is initialized with {} templates from {}"
        message = string.format(self.name, self.overlaps_store.nb_templates, self.init_path)
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
        if self.overlaps_store is not None:
            return self.overlaps_store.nb_templates
        else:
            return 0

    def _guess_output_endpoints(self):
        return

    def _init_temp_window(self):
        self.slice_indices = np.zeros(0, dtype=np.int32)
        self._width = (self.overlaps_store.temporal_width - 1) // 2
        self._2_width = 2 * self._width
        temp_window = np.arange(-self._width, self._width + 1)
        buffer_size = 2 * self.nb_samples
        for idx in range(self.nb_channels):
            self.slice_indices = np.concatenate((self.slice_indices, idx * buffer_size + temp_window))

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

    @property
    def _empty_result(self):

        r = {
            'offset': self.offset,
        }

        return r

    def _reset_result(self):

        self.r = {
            'spike_times': np.zeros(0, dtype=np.int32),
            'amplitudes': np.zeros(0, dtype=np.float32),
            'templates': np.zeros(0, dtype=np.int32),
            'offset': self.offset,
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
        waveforms = np.zeros((self.overlaps_store._nb_elements, nb_peaks), dtype=np.float32)
        for k, peak_time_step in enumerate(peak_time_steps):
            waveforms[:, k] = batch[self.slice_indices + peak_time_step]

        return waveforms

    def _fit_chunk(self):

        # Log some information.
        string = "{} fits spikes... ({} templates)"
        message = string.format(self.name_and_counter, self.nb_templates)
        self.log.debug(message)

        # Reset result.
        self._reset_result()

        # Compute the number of peaks in the current chunk.
        is_in_work_area = np.logical_and(
            self.work_area_start <= self.p,
            self.p < self.work_area_end
        )
        nb_peaks = np.count_nonzero(is_in_work_area)
        peaks = self.p[is_in_work_area]

        # If there is at least one peak in the work area...
        if 0 < nb_peaks:

            # Extract waveforms from buffer.
            waveforms = self._extract_waveforms(peaks)

            # Compute the scalar products between waveforms and templates.
            scalar_products = self.overlaps_store.dot(waveforms)

            # Initialize the failure counter of each peak.
            nb_failures = np.zeros(nb_peaks, dtype=np.int32)
            # Initialize the matching matrix.
            mask = np.ones((self.overlaps_store.nb_templates, nb_peaks), dtype=np.int32)
            # Filter scalar products of the first component of each template.
            sub_b = scalar_products[:self.overlaps_store.nb_templates, :]

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
                    best_amplitude = sub_b[best_template_index, peak_index] / self.overlaps_store._nb_elements
                    if self.overlaps_store.two_components:
                        best_scalar_product = scalar_products[best_template_index + self.nb_templates, peak_index]
                        best_amplitude_2 = best_scalar_product / self.overlaps_store._nb_elements

                    # Compute the best normalized amplitude.
                    best_amplitude_ = best_amplitude / self.overlaps_store.norms['1'][best_template_index]
                    if self.overlaps_store.two_components:
                        best_amplitude_2_ = best_amplitude_2 / self.overlaps_store.norms['2'][best_template_index]
                        _ = best_amplitude_2_  # TODO complete.

                    # Verify amplitude constraint.
                    a_min = self.overlaps_store.amplitudes[best_template_index, 0]
                    a_max = self.overlaps_store.amplitudes[best_template_index, 1]

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
                        indices = np.zeros((self.overlaps_store._overlap_size, len(ytmp)), dtype=np.int32)
                        indices[ytmp, np.arange(len(ytmp))] = 1

                        tmp1_ = self.overlaps_store.get_overlaps(best_template_index, 'first_component')
                        tmp1 = tmp1_.multiply(-best_amplitude).dot(indices)
                        scalar_products[:, is_neighbor[0, :]] += tmp1

                        if self.overlaps_store.two_components:
                            tmp2_ = self.overlaps_store.get_overlaps(best_template_index, 'second_component')
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
            keys = ['spike_times', 'amplitudes', 'templates']
            if self.with_rejected_times:
                keys += ['rejected_times', 'rejected_amplitudes']
            # # Keep only spikes in the result area.
            is_in_result = np.logical_and(
                self.result_area_start <= self.r['spike_times'],
                self.r['spike_times'] < self.result_area_end
            )
            for key in keys:
                self.r[key] = self.r[key][is_in_result]
            # # Sort spike.
            indices = np.argsort(self.r['spike_times'])
            for key in keys:
                self.r[key] = self.r[key][indices]
            # # Modify spike time reference.
            self.r['spike_times'] = self.r['spike_times'] - self.nb_samples

            # Log fitting result.
            nb_spike_times = len(self.r['spike_times'])
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

    @property
    def nb_buffers(self):

        return self.x.shape[0] / self.nb_samples

    @property
    def result_area_start(self):

        return (self.nb_buffers - 1) * self.nb_samples - self.nb_samples / 2

    @property
    def result_area_end(self):

        return (self.nb_buffers - 1) * self.nb_samples + self.nb_samples / 2

    @property
    def work_area_start(self):

        return self.result_area_start - self._width

    @property
    def work_area_end(self):

        return self.result_area_end + self._width

    @property
    def buffer_id(self):

        return self.counter * self._nb_fitters + self._fitter_id

    @property
    def first_buffer_id(self):
        return self.buffer_id - 1

    @property
    def offset(self):

        return self.first_buffer_id * self.nb_samples + self.result_area_start

    def _collect_data(self, shift=0):

        k = (self.nb_buffers - 1) + shift

        self.x[k * self.nb_samples:(k + 1) * self.nb_samples, :] = \
            self.get_input('data').receive(blocking=True)

        return

    def _handle_peaks(self, peaks):

        p = self.nb_samples + self._merge_peaks(peaks)
        self.p = self.p - self.nb_samples
        self.p = self.p[0 <= self.p]
        self.p = np.concatenate((self.p, p))

        return

    def _collect_peaks(self, shift=0):

        if self.is_active:
            peaks = self.get_input('peaks').receive(blocking=True)
            self._handle_peaks(peaks)
        else:
            peaks = self.inputs['peaks'].receive(blocking=False)
            if peaks is None:
                self.p = None
            else:
                p = self.nb_samples + self._merge_peaks(peaks)
                self.p = p
                # Synchronize peak reception.
                while not self._sync_buffer(peaks, self.nb_samples, nb_parallel_blocks=self._nb_fitters,
                                            parallel_block_id=self._fitter_id, shift=shift):
                    peaks = self.inputs['peaks'].receive(blocking=True)
                    self._handle_peaks(peaks)
                # Set active mode.
                self._set_active_mode()

        return

    def _process(self):

        # First, collect all the buffers we need.
        # # Prepare everything to collect buffers.
        if self.counter == 0:
            # Initialize 'self.x'.
            shape = (2 * self.nb_samples, self.nb_channels)
            self.x = np.zeros(shape, dtype=np.float32)
        elif self._nb_fitters == 1:
            # Copy the end of 'self.x' at its beginning.
            self.x[0 * self.nb_samples:1 * self.nb_samples, :] = \
                self.x[1 * self.nb_samples:2 * self.nb_samples, :]
        else:
            pass
        # # Collect precedent data and peaks buffers.
        if self._nb_fitters > 1 and not(self.counter == 0 and self._fitter_id == 0):
            self._collect_data(shift=-1)
            self._collect_peaks(shift=-1)
        # # Collect current data and peaks buffers.
        self._collect_data(shift=0)
        self._collect_peaks(shift=0)
        # # Collect current updater buffer.
        updater = self.inputs['updater'].receive(blocking=False, discarding_eoc=self.discarding_eoc_from_updater)

        if updater is not None:

            self._measure_time('update_start', frequency=1)

            # Create the template dictionary if necessary.
            if self.overlaps_store is None:
                self.template_store = TemplateStore(updater['templates_file'], 'r')
                self.overlaps_store = OverlapsDictionary(self.template_store)
                self._init_temp_window()
            else:
                self.overlaps_store.update(updater['indices'])
            self.overlaps_store.clear_overlaps()

            self._measure_time('update_end', frequency=1)

        if self.p is not None:

            if self.nb_templates > 0:

                self._measure_time('start', frequency=100)

                self._fit_chunk()
                self.get_output('spikes').send(self.r)

                self._measure_time('end', frequency=100)

            elif self._nb_fitters > 1:

                self.get_output('spikes').send(self._empty_result)

        elif self._nb_fitters > 1:

            self.get_output('spikes').send(self._empty_result)

        return

    def _introspect(self):
        """Introspection."""

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self._nb_fitters * self.nb_samples) / self.sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return
