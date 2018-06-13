import numpy as np
from circusort.block.block import Block
from circusort.obj.template_store import TemplateStore

class Meta_merger(Block):
    """Meta merger block

    Performs high level meta merging on Template based on shapes and activities

    """
    # TODO complete docstring.

    name = "Meta merger"

    params = {
        'templates_init_path': None,
        'max_delay': 50,
        'bin_size': 2,
        'lag': 5,
        'sampling_rate': 20000
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)


        # The following lines are useful to avoid some PyCharm's warning.
        self.templates_init_path = self.templates_init_path

        self.add_input('updater')
        self.add_output('spikes', 'dict')
        self._data = {}

    def _cross_corr(self, spike_1, spike_2):

        size = 2 * self.max_delay + 1
        x_cc = np.zeros(size, dtype=np.float32)
        control = 0

        if (len(spike_1) > 0) and (len(spike_2) > 0):

            t1b = np.unique(np.round(spike_1 / self.bin_size))
            t2b = np.unique(np.round(spike_2 / self.bin_size))

            for d in xrange(size):
                x_cc[d] += len(np.intersect1d(t1b, t2b + d - self.max_delay, assume_unique=True))

            x_cc /= self.nb_bins
            control = len(spike_1) * len(spike_2) / float((self.nb_bins ** 2))

        return x_cc * 1e6, control * 1e6

    def _process(self):

        batch = self.input['spikes'].receive(blocking=False)
        updater = self.inputs['updater'].receive(blocking=False, discarding_eoc=self.discarding_eoc_from_updater)

        if updater is not None:

            self._measure_time('update_start', frequency=1)

            indices = updater.get('indices', None)
            # Create the template dictionary if necessary.
            if self._template_store is None:
                self._template_store = TemplateStore(updater['template_store'], mode='r')
                similarities = self._template_store.similarities

        if batch is not None:

            self._measure_time('start', frequency=100)

            offset = batch.pop('offset')
            for key in batch:
                if key in ['spike_times']:
                    spikes = np.array(batch[key]).astype(np.int32)
                    spikes += offset
                if key in ['templates']:
                    templates = np.array(batch[key]).astype(np.int32)

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
