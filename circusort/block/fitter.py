import numpy as np
import os

from circusort.block.block import Block
from circusort.obj.template_store import TemplateStore
from circusort.obj.overlaps_store import OverlapsStore


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
        'templates_init_path': None,
        'overlaps_init_path': None,
        'with_rejected_times': False,
        'sampling_rate': 20e+3,
        'discarding_eoc_from_updater': False,
        '_nb_fitters': 1,
        '_fitter_id': 0,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        # The following lines are useful to avoid some PyCharm's warnings.
        self.templates_init_path = self.templates_init_path
        self.overlaps_init_path = self.overlaps_init_path
        self.with_rejected_times = self.with_rejected_times
        self.sampling_rate = self.sampling_rate
        self.discarding_eoc_from_updater = self.discarding_eoc_from_updater
        self._nb_fitters = self._nb_fitters
        self._fitter_id = self._fitter_id

        self.add_input('updater', structure='dict')
        self.add_input('data', structure='dict')
        self.add_input('peaks', structure='dict')
        self.add_output('spikes', structure='dict')

        self._nb_channels = None
        self._nb_samples = None
        self._number = None

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

        self._template_store = TemplateStore(self.templates_init_path, mode='r')

        # Log info message.
        string = "{} is initialized with {} templates from {}"
        message = string.format(self.name, self._template_store.nb_templates, self.templates_init_path)
        self.log.info(message)

        self._overlaps_store = OverlapsStore(template_store=self._template_store, path=self.overlaps_init_path,
                                             fitting_mode=True)

        # Precompute all the overlaps.
        if self.overlaps_init_path is None:
            # Log info.
            string = "{} precomputes all the overlaps..."
            message = string.format(self.name)
            self.log.info(message)
            # Precompute all the overlaps.
            self._overlaps_store.compute_overlaps()
        else:
            if os.path.isfile(self.overlaps_init_path):
                # Log info.
                string = "{} loads precomputed overlaps from {}..."
                message = string.format(self.name, self.overlaps_init_path)
                self.log.info(message)
                # Load precomputed overlaps.
                self._overlaps_store.load_overlaps(self.overlaps_init_path)
            else:
                # Log info.
                string = "{} precomputes and saves all the overlaps in {}..."
                message = string.format(self.name, self.overlaps_init_path)
                self.log.info(message)
                # Precompute and save all the overlaps.
                self._overlaps_store.compute_overlaps()
                self._overlaps_store.save_overlaps(self.overlaps_init_path)

        return

    @property
    def nb_templates(self):
        if self._overlaps_store is not None:
            return self._overlaps_store.nb_templates
        else:
            return 0

    def _configure_input_parameters(self, nb_channels=None, nb_samples=None, **kwargs):

        if nb_channels is not None:
            self._nb_channels = nb_channels
        if nb_samples is not None:
            self._nb_samples = nb_samples

        return

    def _update_initialization(self):

        if self.templates_init_path is not None:
            self._init_temp_window()

        return

    def _init_temp_window(self):
        # TODO add docstring.

        self.slice_indices = np.zeros(0, dtype=np.int32)
        self._width = (self._overlaps_store.temporal_width - 1) // 2
        self._2_width = 2 * self._width
        temp_window = np.arange(-self._width, self._width + 1)
        buffer_size = 2 * self._nb_samples
        for idx in range(self._nb_channels):
            self.slice_indices = np.concatenate((self.slice_indices, idx * buffer_size + temp_window))

        # Log debug message.
        string = "{} initializes slice indices: {}"
        message = string.format(self.name, self.slice_indices)
        self.log.debug(message)

        return

    def _is_valid(self, peak_step):
        # TODO add docstring.

        i_min = self._width
        i_max = self._nb_samples - self._width
        is_valid = (i_min <= peak_step) & (peak_step < i_max)

        return is_valid

    def _get_all_valid_peaks(self, peak_steps):
        # TODO add docstring.

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
        # TODO add docstring.

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
        waveforms = np.zeros((self._overlaps_store.nb_elements, nb_peaks), dtype=np.float32)
        for k, peak_time_step in enumerate(peak_time_steps):
            waveforms[:, k] = batch[self.slice_indices + peak_time_step]

        return waveforms

    def _fit_chunk(self, verbose=False, timing=False):
        # TODO add docstring.

        if verbose:
            # Log debug message.
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

        if verbose:
            # Log debug message.
            string = "{} has {} peaks in the work area among {} peaks"
            message = string.format(self.name, nb_peaks, len(self.p))
            self.log.debug(message)

        if verbose:
            # Log debug message.
            string = "{} peaks: {}"
            message = string.format(self.name, peaks)
            self.log.debug(message)

        # If there is at least one peak in the work area...
        if 0 < nb_peaks:

            # Extract waveforms from buffer.
            waveforms = self._extract_waveforms(peaks)

            if timing:
                self._measure_time('scalar_products_start', period=10)
            # Compute the scalar products between waveforms and templates.
            scalar_products = self._overlaps_store.dot(waveforms)
            if timing:
                self._measure_time('scalar_products_end', period=10)

            # Initialize the failure counter of each peak.
            nb_failures = np.zeros(nb_peaks, dtype=np.int32)
            # Initialize the matching matrix.
            mask = np.ones((self._overlaps_store.nb_templates, nb_peaks), dtype=np.int32)
            # Filter scalar products of the first component of each template.
            sub_b = scalar_products[:self._overlaps_store.nb_templates, :]

            if verbose:
                # Log debug message.
                string = "{} buffer offset: {}"
                message = string.format(self.name, self.offset)
                self.log.debug(message)

            if timing:
                self._measure_time('while_loop_start', period=10)
            # TODO rewrite condition according to the 3 last lines of the nested while loop.
            # while not np.all(nb_failures == self.max_nb_trials):
            while np.mean(nb_failures) < self.nb_chances:

                # Set scalar products of tested matchings to zero.
                data = sub_b * mask
                # Sort peaks by decreasing highest scalar product with all the templates.
                peak_indices = np.argsort(np.max(data, axis=0))[::-1]
                # TODO remove peaks with scalar products equal to zero?
                # TODO consider the absolute values of the scalar products?

                if timing:
                    self._measure_time('for_loop_start', period=10)
                for peak_index in peak_indices:

                    if timing:
                        self._measure_time('for_loop_preamble_start', period=10)
                    # Find the best template.
                    peak_scalar_products = np.take(sub_b, peak_index, axis=1)
                    best_template_index = np.argmax(peak_scalar_products, axis=0)

                    # Compute the best amplitude.
                    best_amplitude = sub_b[best_template_index, peak_index] / self._overlaps_store.nb_elements
                    if self._overlaps_store.two_components:
                        best_scalar_product = scalar_products[best_template_index + self.nb_templates, peak_index]
                        best_amplitude_2 = best_scalar_product / self._overlaps_store.nb_elements
                    else:
                        best_amplitude_2 = None

                    # Compute the best normalized amplitude.
                    best_amplitude_ = best_amplitude / self._overlaps_store.norms['1'][best_template_index]
                    if self._overlaps_store.two_components:
                        best_amplitude_2_ = best_amplitude_2 / self._overlaps_store.norms['2'][best_template_index]
                        _ = best_amplitude_2_  # TODO complete.

                    # Verify amplitude constraint.
                    a_min = self._overlaps_store.amplitudes[best_template_index, 0]
                    a_max = self._overlaps_store.amplitudes[best_template_index, 1]
                    if timing:
                        self._measure_time('for_loop_preamble_end', period=10)

                    if timing:
                        self._measure_time('for_loop_process_start', period=10)
                    if (a_min <= best_amplitude_) & (best_amplitude_ <= a_max):
                        if verbose:
                            # Log debug message.
                            string = "{} processes (p {}, t {}) -> (a {}, keep)"
                            message = string.format(self.name, peak_index, best_template_index, best_amplitude)
                            self.log.debug(message)
                        if timing:
                            self._measure_time('for_loop_accept_start', period=10)
                        # Keep the matching.
                        peak_time_step = peaks[peak_index]
                        # # Compute the neighboring peaks.
                        # # TODO use this definition of `is_neighbor` instead of the other.
                        # is_neighbor = np.abs(peaks - peak_index) <= 2 * self._width

                        if timing:
                            self._measure_time('for_loop_update_start', period=10)
                        if timing:
                            self._measure_time('for_loop_update_1_start', period=10)
                        # Update scalar products.
                        # TODO simplify the following 11 lines.
                        tmp = np.dot(np.ones((1, 1), dtype=np.int32), np.reshape(peaks, (1, nb_peaks)))
                        tmp -= np.array([[peak_time_step]])
                        is_neighbor = np.abs(tmp) <= self._2_width
                        ytmp = tmp[0, is_neighbor[0, :]] + self._2_width
                        indices = np.zeros((self._overlaps_store.size, len(ytmp)), dtype=np.int32)
                        indices[ytmp, np.arange(len(ytmp))] = 1
                        if timing:
                            self._measure_time('for_loop_update_1_end', period=10)

                        if timing:
                            self._measure_time('for_loop_update_2_start', period=10)
                        if timing:
                            self._measure_time('for_loop_overlaps_start', period=10)
                        tmp1_ = self._overlaps_store.get_overlaps(best_template_index, '1')
                        if timing:
                            self._measure_time('for_loop_overlaps_end', period=10)
                        tmp1 = tmp1_.multiply(-best_amplitude).dot(indices)
                        scalar_products[:, is_neighbor[0, :]] += tmp1
                        if timing:
                            self._measure_time('for_loop_update_2_end', period=10)

                        if self._overlaps_store.two_components:
                            tmp2_ = self._overlaps_store.get_overlaps(best_template_index, '2')
                            tmp2 = tmp2_.multiply(-best_amplitude_2).dot(indices)
                            scalar_products[:, is_neighbor[0, :]] += tmp2
                        if timing:
                            self._measure_time('for_loop_update_end', period=10)

                        if timing:
                            self._measure_time('for_loop_concatenate_start', period=10)
                        # Add matching to the result.
                        self.r['spike_times'] = np.concatenate((self.r['spike_times'], [peak_time_step]))
                        self.r['amplitudes'] = np.concatenate((self.r['amplitudes'], [best_amplitude_]))
                        self.r['templates'] = np.concatenate((self.r['templates'], [best_template_index]))
                        if timing:
                            self._measure_time('for_loop_concatenate_end', period=10)
                        # Mark current matching as tried.
                        mask[best_template_index, peak_index] = 0
                        if timing:
                            self._measure_time('for_loop_accept_end', period=10)
                    else:
                        if verbose:
                            # Log debug message.
                            string = "{} processes (p {}, t {}) -> (a {}, reject)"
                            message = string.format(self.name, peak_index, best_template_index, best_amplitude)
                            self.log.debug(message)
                        if timing:
                            self._measure_time('for_loop_reject_start', period=10)
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
                        if timing:
                            self._measure_time('for_loop_reject_end', period=10)
                    if timing:
                        self._measure_time('for_loop_process_end', period=10)
                if timing:
                    self._measure_time('for_loop_end', period=10)
            if timing:
                self._measure_time('while_loop_end', period=10)

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
            self.r['spike_times'] = self.r['spike_times'] - self._nb_samples

            if verbose:
                # Log debug message.
                nb_spike_times = len(self.r['spike_times'])
                if nb_spike_times > 0:
                    string = "{} fitted {} spikes ({} templates)"
                    message = string.format(self.name_and_counter, nb_spike_times, self.nb_templates)
                else:
                    string = "{} fitted no spikes ({} templates)"
                    message = string.format(self.name_and_counter, self.nb_templates)
                self.log.debug(message)

        else:  # i.e. nb_peaks == 0

            if verbose:
                # Log debug message.
                string = "{} can't fit spikes ({} templates)"
                message = string.format(self.name_and_counter, self.nb_templates)
                self.log.debug(message)

        return

    @staticmethod
    def _merge_peaks(peaks):
        """Merge positive and negative peaks from all the channels."""
        # TODO complete docstring.

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

        return self.x.shape[0] / self._nb_samples

    @property
    def result_area_start(self):

        return (self.nb_buffers - 1) * self._nb_samples - self._nb_samples / 2

    @property
    def result_area_end(self):

        return (self.nb_buffers - 1) * self._nb_samples + self._nb_samples / 2

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

        # TODO check if the comment fix the "offset bug".
        return self.buffer_id  # - 1

    @property
    def offset(self):

        # TODO check if the comment fix the "offset bug".
        return self.first_buffer_id * self._nb_samples  # + self.result_area_start

    def _collect_data(self, shift=0):
        # TODO add docstring.

        k = (self.nb_buffers - 1) + shift

        data_packet = self.get_input('data').receive(blocking=True)
        self._number = data_packet['number']
        self.x[k * self._nb_samples:(k + 1) * self._nb_samples, :] = data_packet['payload']

        return

    def _handle_peaks(self, peaks):
        # TODO add docstring.

        p = self._nb_samples + self._merge_peaks(peaks)
        self.p = self.p - self._nb_samples
        self.p = self.p[0 <= self.p]
        self.p = np.concatenate((self.p, p))

        return

    def _collect_peaks(self, shift=0, verbose=False):
        # TODO add docstring.

        if self.is_active:
            peaks_packet = self.get_input('peaks').receive(blocking=True)
            peaks = peaks_packet['payload']
            self._handle_peaks(peaks)
            if verbose:
                # Log debug message.
                string = "{} collects peaks {} (shift {}, reg)"
                message = string.format(self.name, peaks['offset'], shift)
                self.log.debug(message)
        else:
            peaks_packet = self.get_input('peaks').receive(blocking=False)
            peaks = peaks_packet['payload'] if peaks_packet is not None else None
            if peaks is None:
                self.p = None
            else:
                p = self._nb_samples + self._merge_peaks(peaks)
                self.p = p
                if verbose:
                    # Log debug message.
                    string = "{} collects peaks {} (shift {}, init)"
                    message = string.format(self.name, peaks['offset'], shift)
                    self.log.debug(message)
                    # Log debug message.
                    string = "{} synchronizes peaks ({}, {}, {}, {}, {})"
                    message = string.format(self.name, self._nb_samples, self._nb_fitters,
                                            self._fitter_id, shift, self.counter)
                    self.log.debug(message)
                # Synchronize peak reception.
                while not self._sync_buffer(peaks, self._nb_samples, nb_parallel_blocks=self._nb_fitters,
                                            parallel_block_id=self._fitter_id, shift=shift):
                    peaks_packet = self.get_input('peaks').receive(blocking=True)
                    peaks = peaks_packet['payload']
                    self._handle_peaks(peaks)
                    if verbose:
                        # Log debug message.
                        string = "{} collects peaks {} (shift {}, sync)"
                        message = string.format(self.name, peaks['offset'], shift)
                        self.log.debug(message)
                # Set active mode.
                self._set_active_mode()

        return

    def _process(self, verbose=False, timing=False):
        # TODO add docstring.

        if timing:
            self._measure_time('preamble_start', period=10)

        # First, collect all the buffers we need.
        # # Prepare everything to collect buffers.
        if self.counter == 0:
            # Initialize 'self.x'.
            shape = (2 * self._nb_samples, self._nb_channels)
            self.x = np.zeros(shape, dtype=np.float32)
        elif self._nb_fitters == 1:
            # Copy the end of 'self.x' at its beginning.
            self.x[0 * self._nb_samples:1 * self._nb_samples, :] = \
                self.x[1 * self._nb_samples:2 * self._nb_samples, :]
        else:
            pass
        # # Collect precedent data and peaks buffers.
        if self._nb_fitters > 1 and not(self.counter == 0 and self._fitter_id == 0):
            self._collect_data(shift=-1)
            self._collect_peaks(shift=-1, verbose=verbose)
        # # Collect current data and peaks buffers.
        self._collect_data(shift=0)
        self._collect_peaks(shift=0, verbose=verbose)
        # # Collect current updater buffer.
        updater_packet = self.get_input('updater').receive(blocking=False,
                                                           discarding_eoc=self.discarding_eoc_from_updater)
        updater = updater_packet['payload'] if updater_packet is not None else None

        if timing:
            self._measure_time('preamble_end', period=10)

        if updater is not None:

            self._measure_time('update_start', period=1)

            while updater is not None:

                # Log debug message.
                string = "{} modifies template and overlap stores."
                message = string.format(self.name)
                self.log.debug(message)

                # Create the template dictionary if necessary.
                indices = updater.get('indices', None)
                if self._overlaps_store is None:
                    # Log debug message.
                    string = "{} will initialize template and overlap stores ({}, {})"
                    message = string.format(self.name, updater['template_store'], updater['overlaps']['path'])
                    self.log.debug(message)
                    # Create template store.
                    self._template_store = TemplateStore(updater['template_store'], mode='r')
                    # Create the overlaps store.
                    self._overlaps_store = OverlapsStore(template_store=self._template_store,
                                                         path=updater['overlaps']['path'], fitting_mode=True)
                    self._init_temp_window()
                    # Log debug message.
                    string = "{} initializes template and overlap stores ({}, {})"
                    message = string.format(self.name, updater['template_store'], updater['overlaps']['path'])
                    self.log.debug(message)
                else:
                    # Log debug message.
                    string = "{} will update template and overlap stores"
                    message = string.format(self.name)
                    self.log.debug(message)
                    # Update template and overlap stores.
                    laziness = updater['overlaps']['path'] is None
                    self._overlaps_store.update(indices, laziness=laziness)
                    # Log debug message.
                    string = "{} updates template and overlap stores"
                    message = string.format(self.name)
                    self.log.debug(message)

                # Log debug message.
                string = "{} modified template and overlap stores."
                message = string.format(self.name)
                self.log.debug(message)

                updater_packet = self.get_input('updater').receive(blocking=False,
                                                                   discarding_eoc=self.discarding_eoc_from_updater)
                updater = updater_packet['payload'] if updater_packet is not None else None

            self._measure_time('update_end', period=1)

        if self.p is not None:

            if self.nb_templates > 0:

                self._measure_time('start')

                if timing:
                    self._measure_time('fit_start', period=10)
                self._fit_chunk(verbose=verbose, timing=timing)
                if timing:
                    self._measure_time('fit_end', period=10)
                if timing:
                    self._measure_time('output_start', period=10)
                packet = {
                    'number': self._number,
                    'payload': self.r,
                }
                self.get_output('spikes').send(packet)
                if timing:
                    self._measure_time('output_end', period=10)

                self._measure_time('end')

            elif self._nb_fitters > 1:

                packet = {
                    'number': self._number,
                    'payload': self._empty_result,
                }
                self.get_output('spikes').send(packet)

        elif self._nb_fitters > 1:

            packet = {
                'number': self._number,
                'payload': self._empty_result,
            }
            self.get_output('spikes').send(packet)

        return

    def _introspect(self):
        """Introspection."""
        # TODO complete docstring.

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self._nb_fitters * self._nb_samples) / self.sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        # Log info message.
        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return
