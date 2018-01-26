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
        self.add_output('data')

        # Lines useful to remove PyCharm warnings.
        self.data_path = "/tmp/input.raw" if self.data_path is None else self.data_path
        self.dtype = 'float32' if self.dtype is None else self.dtype
        self.nb_channels = 10 if self.nb_channels is None else self.nb_channels
        self.nb_samples = None if self.nb_samples is None else self.nb_samples
        self.sampling_rate = None if self.sampling_rate is None else self.sampling_rate
        self.is_realistic = True if self.is_realistic is None else self.is_realistic

        self.output_dtype = 'float32'
        self.quantum_size = 0.1042  # µV / AD
        self.quantum_offset = float(np.iinfo('int16').min)

        self.buffer_rate = float(self.nb_samples) / self.sampling_rate

    def _initialize(self):
        """Initialization of the processing block."""

        data = np.memmap(self.data_path, dtype=self.dtype, mode='r')
        self.shape = (data.size / self.nb_channels, self.nb_channels)
        self.output.configure(dtype=self.output_dtype, shape=(self.nb_samples, self.nb_channels))

        return

    def _process(self):
        """Process one buffer of data."""

        # TODO check if we need a background thread.

        self._measure_time(label='start', frequency=100)

        # Read data from the file on disk.
        data = np.memmap(self.data_path, dtype=self.dtype, mode='r', shape=self.shape)

        i_min = self.nb_samples * self.counter
        i_max = self.nb_samples * (self.counter + 1)

        if i_max <= self.shape[0]:
            # Get chunk.
            chunk = data[i_min:i_max, :]
            # Dequantize chunk.
            if self.dtype == 'float32':
                pass
            elif self.dtype == 'int16':
                chunk = chunk.astype(self.output_dtype)
                chunk *= self.quantum_size
            elif self.dtype == 'uint16':
                chunk = chunk.astype(self.output_dtype)
                chunk += self.quantum_offset
                chunk *= self.quantum_size
            else:
                chunk = chunk.astype(self.output_dtype)
            # Send chunk.
            if self.is_realistic:
                # Simulate duration between two data acquisitions.
                time.sleep(float(self.nb_samples) / self.sampling_rate)
            self.output.send(chunk)
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

        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return
