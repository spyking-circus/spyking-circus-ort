import numpy as np

from circusort.block.block import Block


__classname__ = "ChannelGrouper"


class ChannelGrouper(Block):
    """Channel grouper.

    Attribute:
        nb_groups: integer
    """

    name = "Channel grouper"

    params = {
        'nb_groups': 1
    }

    def __init__(self, **kwargs):
        """Initialize channel grouper.

        Argument:
            nb_group: integer (optional)
            The number of groups from which data will be gathered.
            The default value is 1.
        """

        Block.__init__(self, **kwargs)

        # The following line is useful to avoid some PyCharm's warning.
        self.nb_groups = self.nb_groups

        for k in range(0, self.nb_groups):
            self.add_input('data_{}'.format(k), structure='dict')
        self.add_output('data', structure='dict')

        self.dtype = None
        self.nb_samples = None
        self.nb_channels = None
        self.sampling_rate = None

        self._result = None

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

        # Compute the number of channels grouped from the input endpoints.
        nb_channels = 0
        for k in range(0, self.nb_groups):
            input_name = 'data_{}'.format(k)
            input_endpoint = self.get_input(input_name)
            input_parameters = input_endpoint.get_input_parameters()
            nb_channels += input_parameters['nb_channels']
        self.configure_input_parameters(nb_channels=nb_channels)

        shape = (self.nb_samples, self.nb_channels)
        self._result = np.zeros(shape, dtype=self.dtype)

        # Log debug message.
        string = "{} updated initialization (nb_channels={})"
        message = string.format(self.name_and_counter, self.nb_channels)
        self.log.debug(message)

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

        # Receive input packets.
        packets = {}
        for k in range(0, self.nb_groups):
            input_name = 'data_{}'.format(k)
            packets[k] = self.get_input(input_name).receive()

        self._measure_time('start')

        # Unpack input packets.
        number = None
        for k in range(0, self.nb_groups):
            number = packets[k]['number']  # TODO check that all the number are the same.
            batch = packets[k]['payload']
            self._result[:, k::self.nb_groups] = batch

        # Send output packet.
        packet = {
            'number': number,
            'payload': self._result,
        }
        self.output.send(packet)

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
