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

        return

    def _process(self):

        number = None

        for k in range(0, self.nb_groups):
            input_name = 'data_{}'.format(k)
            packet = self.get_input(input_name).receive()
            number = packet['number']  # TODO check that all the number are the same.
            batch = packet['payload']
            self._result[:, k::self.nb_groups] = batch

        packet = {
            'number': number,
            'payload': self._result,
        }
        self.output.send(packet)

        return
