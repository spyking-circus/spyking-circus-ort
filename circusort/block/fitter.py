# import matplotlib.pyplot as plt
import numpy as np
import os

from circusort.block.block import Block
from circusort.utils.fitter import largest_indices
from circusort.obj.template_store import TemplateStore
from circusort.obj.overlaps_store import OverlapsStore


__classname__ = "Fitter"


class Fitter(Block):
    """Fitter

    Attributes:
        templates_init_path: none | string (optional)
            Path to the location used to load templates to initialize the
            dictionary of templates. If equal to None, this dictionary will
            start empty. The default value is None.
        overlaps_init_path: none | string (optional)
            Path to the location used to load the overlaps to initialize the
            overlap store.
            The default value is None.
        with_updater: boolean (optional)
            The default value is True.
        with_rejected_times: boolean (optional)
            The default value is False.
        sampling_rate: float (optional)
            The default value is 20e+3.
        discarding_eoc_from_updater: boolean (optional)
            The default value is False.
    """

    name = "Fitter"

    params = {
        'templates_init_path': None,
        'overlaps_init_path': None,
        'with_updater': True,
        'with_rejected_times': False,
        'sampling_rate': 20e+3,
        'discarding_eoc_from_updater': False,
        '_nb_fitters': 1,
        '_fitter_id': 0,
    }

    def __init__(self, **kwargs):
        """Initialize fitter

        Arguments:
            templates_init_path: string (optional)
            overlaps_init_path: string (optional)
            with_updater: boolean (optional)
            with_rejected_times: boolean (optional)
            sampling_rate: float (optional)
            discarding_eoc_from_updater: boolean (optional)
            _nb_fitters: integer (optional)
            _fitter_id: integer (optional)
        """

        Block.__init__(self, **kwargs)

        # The following lines are useful to avoid some PyCharm's warnings.
        self.templates_init_path = self.templates_init_path
        self.overlaps_init_path = self.overlaps_init_path
        self.with_updater = self.with_updater
        self.with_rejected_times = self.with_rejected_times
        self.sampling_rate = self.sampling_rate
        self.discarding_eoc_from_updater = self.discarding_eoc_from_updater
        self._nb_fitters = self._nb_fitters
        self._fitter_id = self._fitter_id

        # Initialize private attributes.
        self._template_store = None
        self._overlaps_store = None

        if self.with_updater:
            self.add_input('updater', structure='dict')
        self.add_input('data', structure='dict')
        self.add_input('peaks', structure='dict')
        self.add_output('spikes', structure='dict')

        self._nb_channels = None
        self._nb_samples = None
        self._number = None

    def _initialize(self):

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

        # Log info message.
        string = "{} is initialized with precomputed overlaps from {}"
        message = string.format(self.name, self.overlaps_init_path)
        self.log.info(message)

        return

    @property
    def nb_templates(self):

        if self._overlaps_store is not None:
            nb_templates = self._overlaps_store.nb_templates
        else:
            nb_templates = 0

        return nb_templates

    @property
    def min_scalar_product(self):
        return np.min(self._overlaps_store.amplitudes[:, 0] * self._overlaps_store.norms['1'])    

    @property
    def max_scalar_product(self):
        return np.max(self._overlaps_store.amplitudes[:, 1] * self._overlaps_store.norms['1'])    

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

        self._width = (self._overlaps_store.temporal_width - 1) // 2
        self._2_width = 2 * self._width
        self.temp_window = np.arange(-self._width, self._width + 1)
        buffer_size = 2 * self._nb_samples
        return

    def _is_valid(self, peak_step):

        i_min = self._width
        i_max = self._nb_samples - self._width
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

        Argument:
            peak_time_steps: np.array
                Peak time steps. Array of shape (number of peaks,).

        Return:
            waveforms: np.array
                Extracted waveforms. Array of shape (waveform size, number of peaks), where waveform size is equal to
                number of channels x number of samples.
        """

        waveforms = self.x[peak_time_steps[:, None] + self.temp_window]
        waveforms = waveforms.transpose(2, 1, 0).reshape(self._overlaps_store.nb_elements, len(peak_time_steps))
        return waveforms

    def _fit_chunk(self, verbose=False, timing=False):

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
        if nb_peaks > 0:

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
            
            if verbose:
                # Log debug message.
                string = "{} buffer offset: {}"
                message = string.format(self.name, self.offset)
                self.log.debug(message)

            if timing:
                self._measure_time('while_loop_start', period=10)
            # TODO rewrite condition according to the 3 last lines of the nested while loop.
            # while not np.all(nb_failures == self.max_nb_trials):

            # Verify amplitude constraint.

            min_scalar_products = self._overlaps_store.amplitudes[:, 0][:, np.newaxis]
            max_scalar_products = self._overlaps_store.amplitudes[:, 1][:, np.newaxis]
            min_sps = min_scalar_products * self._overlaps_store.norms['1'][:, np.newaxis]
            max_sps = max_scalar_products * self._overlaps_store.norms['1'][:, np.newaxis]


            # Set scalar products of tested matches to zero.
            data = scalar_products[:self._overlaps_store.nb_templates, :]

            while True:

                is_valid = (data > min_sps)*(data < max_sps)
                valid_indices = np.where(is_valid)

                if len(valid_indices[0]) == 0:
                    break

                best_amplitude = data[is_valid].argmax()
                best_template_index, peak_index = valid_indices[0][best_amplitude_idx], valid_indices[1][best_amplitude_idx]

                # Compute the best normalized amplitude.
                best_amplitude_ = best_amplitude / self._overlaps_store.norms['1'][best_template_index]

                if self._overlaps_store.two_components:
                    best_amplitude_2 = scalar_products[best_template_index + self.nb_templates, peak_index]
                    best_amplitude_2_ = best_amplitude_2 / self._overlaps_store.norms['2'][best_template_index]
                
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
                tmp1 = tmp1_.multiply(-best_amplitude)

                if self._overlaps_store.two_components:
                    tmp1_ += self._overlaps_store.get_overlaps(best_template_index, '2').multiply(-best_amplitude_2)

                to_add = tmp1.toarray()[:, ytmp]
                scalar_products[:, is_neighbor[0, :]] += to_add

                if timing:
                    self._measure_time('for_loop_update_2_end', period=10)

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
                scalar_products[best_template_index, peak_index] = -np.inf

                if timing:
                    self._measure_time('for_loop_accept_end', period=10)

                        
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

        time_steps = set([])
        keys = [
            key
            for key in peaks.keys()
            if key not in ['offset']
        ]
        for key in keys:
            for channel in peaks[key].keys():
                time_steps = time_steps.union(peaks[key][channel])
        time_steps = np.array(list(time_steps), dtype=np.int32)
        time_steps = np.sort(time_steps)

        return time_steps

    @property
    def nb_buffers(self):

        return self.x.shape[0] // self._nb_samples

    @property
    def result_area_start(self):

        return (self.nb_buffers - 1) * self._nb_samples - self._nb_samples // 2

    @property
    def result_area_end(self):

        return (self.nb_buffers - 1) * self._nb_samples + self._nb_samples // 2

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

        k = (self.nb_buffers - 1) + shift

        data_packet = self.get_input('data').receive(blocking=True)
        self._number = data_packet['number']
        self.x[k * self._nb_samples:(k + 1) * self._nb_samples, :] = data_packet['payload']

        return

    def _handle_peaks(self, peaks):

        p = self._nb_samples + self._merge_peaks(peaks)
        self.p = self.p - self._nb_samples
        self.p = self.p[0 <= self.p]
        self.p = np.concatenate((self.p, p))

        return

    def _collect_peaks(self, verbose=False):

        if self.is_active:
            peaks_packet = self.get_input('peaks').receive(blocking=True, number=self._number)
            if peaks_packet is None:
                # This is the last packet (last data packet don't have a corresponding peak packet since the peak
                # detector needs two consecutive data packets to produce one peak packet).
                peaks = {}
            else:
                peaks = peaks_packet['payload']['peaks']
            self._handle_peaks(peaks)
            if verbose:
                # Log debug message.
                string = "{} collects peaks {} (reg)"
                message = string.format(self.name, peaks_packet['payload']['offset'])
                self.log.debug(message)
        else:
            if self.get_input('peaks').has_received():
                peaks_packet = self.get_input('peaks').receive(blocking=True, number=self._number)
                if peaks_packet is not None:
                    peaks = peaks_packet['payload']['peaks']
                    p = self._nb_samples + self._merge_peaks(peaks)
                    self.p = p
                    if verbose:
                        # Log debug message.
                        string = "{} collects peaks {} (init)"
                        message = string.format(self.name, peaks_packet['payload']['offset'])
                        self.log.debug(message)
                    # Set active mode.
                    self._set_active_mode()
                else:
                    self.p = None
            else:
                self.p = None

        return

    def _process(self, verbose=False, timing=False):

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
            self._collect_peaks(verbose=verbose)
        # # Collect current data and peaks buffers.
        self._collect_data(shift=0)
        self._collect_peaks(verbose=verbose)
        # # Collect current updater buffer.
        if self.with_updater:
            updater_packet = self.get_input('updater').receive(
                blocking=False,
                discarding_eoc=self.discarding_eoc_from_updater
            )
            updater = updater_packet['payload'] if updater_packet is not None else None
        else:
            updater = None

        if timing:
            self._measure_time('preamble_end', period=10)

        if updater is not None:

            self._measure_time('update_start', period=1)

            while updater is not None:

                # Log debug message.
                string = "{} modifies template and overlap stores"
                message = string.format(self.name_and_counter)
                self.log.debug(message)

                # Modify template and overlap stores.
                indices = updater.get('indices', None)
                _ = indices  # Discard unused variable.
                if self._template_store is None:
                    # Initialize template and overlap stores.
                    self._template_store = TemplateStore(updater['template_store'], mode='r')
                    self._overlaps_store = OverlapsStore(template_store=self._template_store,
                                                         path=updater['overlaps']['path'], fitting_mode=True)
                    self._init_temp_window()
                    # Log debug message.
                    string = "{} initializes template and overlap stores ({}, {})"
                    message = string.format(self.name_and_counter, updater['template_store'],
                                            updater['overlaps']['path'])
                    self.log.debug(message)

                else:

                    # TODO avoid duplicates in template store and uncomment the 3 following lines.
                    # Update template and overlap stores.
                    laziness = updater['overlaps']['path'] is None
                    self._overlaps_store.update(indices, laziness=laziness)
                    # Log debug message.
                    string = "{} updates template and overlap stores"
                    message = string.format(self.name_and_counter)
                    self.log.debug(message)

                # Log debug message.
                string = "{} modified template and overlap stores"
                message = string.format(self.name_and_counter)
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
