from circusort.block.block import Block


__classname__ = "ChannelDispatcher"


class ChannelDispatcher(Block):
    """Channel dispatcher.

    Attribute:
        nb_groups: integer
            The default value is 1.
    """

    name = "Channel dispatcher"

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

        self.add_input('data', structure='dict')
        for k in range(0, self.nb_groups):
            output_name = 'data_{}'.format(k)
            self.add_output(output_name, structure='dict')

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

        for k in range(0, self.nb_groups):
            output_name = 'data_{}'.format(k)
            output_endpoint = self.get_output(output_name)
            nb_channels = self.nb_channels // self.nb_groups
            if k < self.nb_channels % self.nb_groups:
                nb_channels += 1
            output_endpoint.configure_output_parameters(nb_channels=nb_channels)

        return

    def _get_output_parameters(self):

        params = {
            'dtype': self.dtype,
            'nb_samples': self.nb_samples,
            'sampling_rate': self.sampling_rate,
        }

        return params

    def _process(self):

        input_packet = self.get_input('data').receive()
        number = input_packet['number']
        batch = input_packet['payload']

        for k in range(0, self.nb_groups):
            output_name = 'data_{}'.format(k)
            output_packet = {
                'number': number,
                'payload': batch[:, k::self.nb_groups]
            }
            self.get_output(output_name).send(output_packet)

        return
