# -*- coding: utf-8 -*-

import numpy as np
import time
import warnings
from circusort.io.probe import load_probe

from circusort.block.block import Block


__classname__ = 'Reader'


class Reader(Block):
    """Reader block.

    Attributes:
        data_path: string
        dtype: string
        nb_channels: integer
        nb_samples: integer
        sampling_rate: float
        is_realistic: boolean
        speed_factor: float
        nb_replay: integer
        offset: integer

    See also:
        circusort.block.Block
    """

    name = "File reader"

    params = {
        'data_path': "/tmp/input.raw",
        'dtype': 'float32',
        'nb_channels': 10,
        'nb_samples': 1024,
        'sampling_rate': 20e+3,
        'is_realistic': True,
        'speed_factor': 1.0,
        'nb_replay': 1,
        'offset': 0,
        'probe_path': None
    }

    def __init__(self, **kwargs):
        """Initialization of the object.

        Parameters:
            data_path: string
            dtype: string
            nb_channels: integer
            nb_samples: integer
            sampling_rate: float
            is_realistic: boolean
            speed_factor: float
            nb_replay: integer
            probe_path: string

        See also:
            circusort.block.Block
        """

        Block.__init__(self, **kwargs)
        self.add_output('data', structure='dict')

        # Lines useful to remove PyCharm warnings.
        self.data_path = self.data_path
        self.dtype = self.dtype
        self.nb_channels = self.nb_channels
        self.nb_samples = self.nb_samples
        self.sampling_rate = self.sampling_rate
        self.is_realistic = self.is_realistic
        self.speed_factor = self.speed_factor
        self.nb_replay = self.nb_replay
        self.offset = self.offset
        self.probe_path = self.probe_path
        self._output_dtype = 'float32'
        self._quantum_size = 0.1042  # µV / AD
        self._quantum_offset = float(np.iinfo('int16').min)
        self._buffer_rate = float(self.nb_samples) / self.sampling_rate

        self._absolute_start_time = None
        self._absolute_end_time = None

        if self.probe_path is not None:
            self.probe = load_probe(self.probe_path, logger=self.log)
            # Log info message.
            string = "{} reads the probe layout"
            message = string.format(self.name)
            self.log.info(message)
        else:
            self.probe = None

    @property
    def nb_output_channels(self):
        if self.probe is None:
            return self.nb_channels
        else:
            return self.probe.nb_channels

    def _initialize(self):
        """Initialization of the processing block."""

        data = np.memmap(self.data_path, dtype=self.dtype, offset=self.offset, mode='r')
        self.real_shape = (data.size // self.nb_channels, self.nb_channels)
        self.shape = (self.real_shape[0] * self.nb_replay, self.nb_output_channels)
        self.output.configure(dtype=self._output_dtype, shape=(self.nb_samples, self.nb_output_channels))
        return

    def _get_output_parameters(self):
        """Collect parameters to transmit to output blocks."""

        params = {
            'dtype': self._output_dtype,
            'nb_channels': self.nb_output_channels,
            'nb_samples': self.nb_samples,
            'sampling_rate': self.sampling_rate,
        }

        return params

    def _process(self):
        """Process one buffer of data."""

        # TODO check if we need a background thread.

        self._measure_time(label='start')

        # Initialize start time (if necessary).
        if self._absolute_start_time is None:
            self._absolute_start_time = time.time()

        # Read data from the file on disk.
        data = np.memmap(self.data_path, dtype=self.dtype, mode='r', shape=self.real_shape, offset=self.offset)

        g_min = (self.nb_samples * self.counter)

        if g_min < self.shape[0]:
            # Get chunk.
            i_min = (self.nb_samples * self.counter) % self.real_shape[0]
            i_max = i_min + self.nb_samples

            chunk = data[i_min:i_max, :]
            if self.probe is not None:
                chunk = chunk[:, self.probe.nodes]

            # Repeat last sampling time (if necessary, data buffer incomplete).
            if chunk.shape[0] < self.nb_samples:
                nb_samples = chunk.shape[0]
                nb_missing_samples = self.nb_samples - nb_samples
                indices = np.concatenate((np.arange(0, nb_samples), -1 * np.ones(nb_missing_samples, dtype='int')))
                chunk = chunk[indices, :]
            # Dequantize chunk.
            if self.dtype == 'float32':
                pass
            elif self.dtype == 'int16':
                chunk = chunk.astype(self._output_dtype)
                chunk *= self._quantum_size
            elif self.dtype == 'uint16':
                chunk = chunk.astype(self._output_dtype)
                chunk += self._quantum_offset
                chunk *= self._quantum_size
            else:
                chunk = chunk.astype(self._output_dtype)
            # Prepare output data packet.
            packet = {
                'number': self.counter,
                'payload': chunk,
            }
            # Send output data packet.
            if self.is_realistic:
                # Simulate duration between two data acquisitions.
                expected_relative_output_time = float(self.counter + 1) * float(self.nb_samples) / self.sampling_rate
                expected_relative_output_time /= self.speed_factor
                expected_absolute_output_time = self._absolute_start_time + expected_relative_output_time
                absolute_current_time = time.time()
                try:
                    time.sleep(expected_absolute_output_time - absolute_current_time)
                except ValueError:
                    lag_duration = expected_absolute_output_time - absolute_current_time
                    string = "{} breaks realistic mode (lag: +{} s)"
                    message = string.format(self.name_and_counter, lag_duration)
                    warnings.warn(message)
            self.output.send(packet)
        else:
            # Log debug message.
            self._absolute_end_time = time.time()
            execution_duration = self._absolute_end_time - self._absolute_start_time
            string = "{} executes during {} s"
            message = string.format(self.name_and_counter, execution_duration)
            self.log.debug(message)
            # Stop processing block.
            self.stop_pending = True

        self._measure_time(label='end')

        return

    def _introspect(self):

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self.nb_samples) / self.sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        # Log info message.
        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return
