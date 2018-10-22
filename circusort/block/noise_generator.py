import numpy as np
import time
import warnings

from .block import Block


__classname__ = "NoiseGenerator"


class NoiseGenerator(Block):

    name = "Noise Generator"

    params = {
        'dtype': 'float32',
        'nb_channels': 10,
        'sampling_rate': 20e+3,  # Hz
        'nb_samples': 1024
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        # The following lines are useful to disable some PyCharm warnings.
        self.dtype = self.dtype
        self.nb_channels = self.nb_channels
        self.sampling_rate = self.sampling_rate
        self.nb_samples = self.nb_samples

        self.add_output('data', structure='dict')

        self._number = -1
        self._absolute_start_time = None

    def _initialize(self):

        pass

        return

    def _get_output_parameters(self):

        params = {
            'dtype': self.dtype,
            'nb_samples': self.nb_samples,
            'nb_channels': self.nb_channels,
            'sampling_rate': self.sampling_rate,
        }

        return params

    def _process(self):

        if self._absolute_start_time is None:
            self._absolute_start_time = time.time()

        self._number += 1
        batch = np.random.randn(self.nb_samples, self.nb_channels)
        batch = batch.astype(self.dtype)
        packet = {
            'number': self._number,
            'payload': batch,
        }

        # Simulate duration between two consecutive data acquisitions.
        expected_relative_output_time = float(self._number + 1) * float(self.nb_samples) / self.sampling_rate
        expected_absolute_output_time = self._absolute_start_time + expected_relative_output_time
        absolute_current_time = time.time()
        try:
            time.sleep(expected_absolute_output_time - absolute_current_time)
        except ValueError:
            lag_duration = expected_absolute_output_time - absolute_current_time
            string = "{} breaks realistic mode (lag: +{} s)"
            message = string.format(self.name_and_counter, lag_duration)
            warnings.warn(message)

        self.get_output('data').send(packet)

        return
