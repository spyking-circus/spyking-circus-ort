import numpy as np

from circusort.block.block import Block


__classname__ = "ClusteringDispatcher"


class ClusteringDispatcher(Block):
    """Channel dispatcher.

    Attribute:
        nb_groups: integer
            The default value is 1.
    """

    name = "Clustering dispatcher"

    params = {
        'nb_samples' : 1024
    }

    def __init__(self, **kwargs):
        """Initialize channel dispatcher.
        """

        Block.__init__(self, **kwargs)

        keys =  ['data', 'peaks', 'pcs']

        for key in keys:
            self.add_input(key, structure='dict')
            self.add_output(key, structure='dict')

        self._nb_samples = None

    def _initialize(self):

        return


    def _configure_input_parameters(self, dtype=None, nb_samples=None, nb_channels=None, sampling_rate=None, **kwargs):

        if dtype is not None:
            self.dtype = dtype
        if nb_samples is not None:
            self.nb_samples = nb_samples
        if nb_channels is not None:
            self.nb_channels = nb_channels
        if sampling_rate is not None:
            self.sampling_rate = sampling_rate

    def _update_initialization(self):

        output_name = 'data'
        output_endpoint = self.get_output(output_name)
        output_endpoint.configure_output_parameters(nb_channels=self.nb_channels)

        # Log debug message.
        string = "{} updated initialization"
        message = string.format(self.name_and_counter)
        self.log.debug(message)

        return

    def _get_output_parameters(self):

        params = {
            'dtype': self.dtype,
            'nb_samples': self.nb_samples,
            'sampling_rate': self.sampling_rate,
            'nb_channels' : self.nb_channels
        }

        return params

    def _process(self):

        data_packet = self.get_input('data').receive()
        peaks_paquet = self.get_input('peaks').receive(blocking=False)
        pcs_packet = self.get_input('pcs').receive(blocking=False)

        self._measure_time('start')

        # Send output packets.
        for output_name in self.outputs.keys():
            if output_name == 'data':
                to_send = data_packet
            elif output_name == 'peaks':
                to_send = peaks_paquet
            elif output_name == 'pcs':
                to_send = pcs_packet
            self.get_output(output_name).send(to_send)

        self._measure_time('end')

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
