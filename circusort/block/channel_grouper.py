import numpy as np

from circusort.block.block import Block


__classname__ = "ChannelGrouper"


class ChannelGrouper(Block):
    """Channel grouper.

    Attribute:
        nb_groups: integer
    """

    name = "Channel Grouper"

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
            self.add_input('data_{}'.format(k))
        self.add_output('data')

    @property
    def nb_channels(self):

        shape = 0
        for k in range(0, self.nb_groups):
            input_name = 'data_{}'.format(k)
            shape += self.get_input(input_name).shape[1]

        return shape

    @property
    def nb_samples(self):

        return self.get_input('data_0').shape[0]

    @property
    def dtype(self):

        return self.get_input('data_0').dtype

    def _initialize(self):

        pass

        return

    def _guess_output_endpoints(self):

        try:
            self.output.configure(dtype=self.dtype, shape=(self.nb_samples, self.nb_channels))
            self.result = np.zeros((self.nb_samples, self.nb_channels), dtype=self.dtype)
        except Exception:  # TODO narrow the exception catch.
            message = "Not all input connections have been established!"
            self.log.debug(message)

        return

    def _process(self):

        for k in range(self.nb_groups):
            input_name = 'data_{}'.format(k)
            batch = self.get_input(input_name).receive()
            self.result[:, k::self.nb_groups] = batch

        self.output.send(self.result)

        return
