# -*- coding: utf-8 -*-

import numpy as np
import time

from circusort.block.block import Block


class Reader(Block):
    """Reader block.

    Attributes:
        data_path: string
        dtype: string
        nb_channels: integer
        nb_samples: integer
        sampling_rate: float
        is_realistic: boolean

    See also:
        circusort.block.Block
    """
    # TODO complete docstring.

    name = "File reader"

    params = {
        'data_path': "/tmp/input.raw",
        'dtype': 'float32',
        'nb_channels': 10,
        'nb_samples': 1024,
        'sampling_rate': 20e+3,
        'is_realistic': True,
        'nb_replay': 1,
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

        See also:
            circusort.block.Block
        """
        # TODO complete docstring.

        Block.__init__(self, **kwargs)
        self.add_output('data', structure='dict')

        # Lines useful to remove PyCharm warnings.
        self.data_path = self.data_path
        self.dtype = self.dtype
        self.nb_channels = self.nb_channels
        self.nb_samples = self.nb_samples
        self.sampling_rate = self.sampling_rate
        self.is_realistic = self.is_realistic
        self.nb_replay = self.nb_replay

        self._output_dtype = 'float32'
        self._quantum_size = 0.1042  # µV / AD
        self._quantum_offset = float(np.iinfo('int16').min)
        self._buffer_rate = float(self.nb_samples) / self.sampling_rate

    def _initialize(self):
        """Initialization of the processing block."""

        data = np.memmap(self.data_path, dtype=self.dtype, mode='r')
        self.real_shape = (data.size / self.nb_channels, self.nb_channels)
        self.shape = (self.real_shape[0] * self.nb_replay, self.real_shape[1])
        self.output.configure(dtype=self._output_dtype, shape=(self.nb_samples, self.nb_channels))

        return

    def _get_output_parameters(self):
        """Collect parameters to transmit to output blocks."""

        params = {
            'dtype': self._output_dtype,
            'nb_channels': self.nb_channels,
            'nb_samples': self.nb_samples,
        }

        return params

    def _process(self):
        """Process one buffer of data."""

        # TODO check if we need a background thread.

        self._measure_time(label='start', frequency=100)

        # Read data from the file on disk.
        data = np.memmap(self.data_path, dtype=self.dtype, mode='r', shape=self.real_shape)

        g_min = (self.nb_samples * self.counter)

        if g_min < self.shape[0]:
            # Get chunk.
            i_min = (self.nb_samples * self.counter) % self.real_shape[0]
            i_max = i_min + self.nb_samples

            chunk = data[i_min:i_max, :]
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
                duration = float(self.nb_samples) / self.sampling_rate
                time.sleep(duration)
            self.output.send(packet)
        else:
            # Stop processing block.
            self.stop_pending = True

        self._measure_time(label='end', frequency=100)

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

        # Log info message.
        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return
