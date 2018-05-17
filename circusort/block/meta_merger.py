import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
import numpy as np
import os
import tempfile
import time

from circusort.block.block import Block


class Meta_merger(Block):
    """Meta merger block

    Performs high level meta merging on Template based on shapes and activities

    """
    # TODO complete docstring.

    name = "Spike writer"

    params = {
        'template_path': None
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        # The following lines are useful to avoid some PyCharm's warning.
        self.add_input('spikes')

    def _process(self):

        batch = self.input.receive(blocking=False)

        if batch is not None:

            self._measure_time('start', frequency=100)

            offset = batch.pop('offset')
            if self._mode == 'raw':
                for key in batch:
                    if key in ['spike_times']:
                        to_write = np.array(batch[key]).astype(np.int32)
                        to_write += offset
                    elif key in ['templates']:
                        to_write = np.array(batch[key]).astype(np.int32)
                    elif key in ['amplitudes']:
                        to_write = np.array(batch[key]).astype(np.float32)

            self._measure_time('end', frequency=100)

        return

    def _introspect(self):
        """Introspection of this block for spike writing."""

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
