import numpy as np

from circusort.block.block import Block


__classname__ = "ClusteringDispatcher"


class ClusteringDispatcher(Block):
    """Node to dispathc data for clsutering network

    Attribute:
        nb_groups: integer
            The default value is 1.
    """

    name = "Channel dispatcher"

    def __init__(self, **kwargs):
        """Initialize channel dispatcher.

        Argument:
            nb_groups: integer (optional)
                The number of groups into which data will be dispatch.
                The default value is 1.
        """

        Block.__init__(self, **kwargs)

        # The following line is useful to disable some PyCharm's warning

        self.names = ['data', 'peaks', 'pcs', 'mads']
        for n in self.names:
            self.add_input(n, structure='dict')
            self.add_output(n, structure='dict')

        self.dtype = None
        self.nb_samples = None
        self.nb_channels = None
        self.sampling_rate = None

    def _initialize(self):

        pass

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

        return

    def _update_initialization(self):

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
        }

        return params

    def _process(self):
    
        results = {}

        results['data'] = self.inputs['data'].receive()
        results['mads'] = self.inputs['mads'].receive(blocking=False)
        results['pcs'] = self.inputs['pcs'].receive(blocking=False)
        results['peaks'] = self.get_input('peaks').receive()

        self._measure_time('start')

        # Unpack input packet.
        number = input_packet['number']
        batch = input_packet['payload']

        # Send output packets.
        for name in self._names:
            self.get_output(name).send(results[name]['payload'])

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
