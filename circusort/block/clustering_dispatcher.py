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
        'nb_groups': 1
    }

    def __init__(self, **kwargs):
        """Initialize channel dispatcher.

        Argument:
            nb_groups: integer (optional)
                The number of groups into which data will be dispatch.
                The default value is 1.
        """

        Block.__init__(self, **kwargs)

        # The following line is useful to disable some PyCharm's warning.
        self.nb_groups = self.nb_groups

        keys =  ['data', 'peaks', 'mads', 'pcs']

        for key in keys:
            self.add_input(key, structure='dict')
            self.add_output(key, structure='dict')

        self.dtype = None
        self.nb_samples = None
        self.nb_channels = None
        self.sampling_rate = None

    def _initialize(self):

        pass

        return

    def _configure_input_parameters(self, dtype=None, nb_samples=None, nb_channels=None, sampling_rate=None, **kwargs):

        return

    def _update_initialization(self):

        # Log debug message.
        string = "{} updated initialization"
        message = string.format(self.name_and_counter)
        self.log.debug(message)

        return

    def _get_output_parameters(self):

        params = {
        }

        return params

    def _process(self):

        data_packet = self.get_input('data').receive()
        peaks_paquet = self.get_input('peaks').receive(blocking=False)
        mads_packet = self.get_input('mads').receive(blocking=False)
        pcs_packet = self.get_input('pcs').receive(blocking=False)

        self._measure_time('start')

        # Send output packets.
        for output_name in self.outputs.keys():
            self.get_output(output_name).send(data_packet)
            self.get_output(output_name).send(peaks_paquet)
            self.get_output(output_name).send(mads_packet)
            self.get_output(output_name).send(pcs_packet)

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
