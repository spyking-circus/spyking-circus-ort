import numpy as np

from circusort.block.block import Block


__classname__ = "ChannelDispatcher"


class ChannelDispatcher(Block):
    """Channel dispatcher.

    Attribute:
        nb_groups: integer
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

        self.add_input('data')
        for k in range(0, self.nb_groups):
            output_name = 'data_{}'.format(k)
            self.add_output(output_name)

    @property
    def nb_channels(self):

        return self.input.shape[1]

    @property
    def nb_samples(self):

        return self.input.shape[0]

    def _initialize(self):

        pass

        return

    def _guess_output_endpoints(self):

        for k in range(0, self.nb_groups):
            shape = len(np.arange(k, self.nb_channels, self.nb_groups))
            output_name = 'data_{}'.format(k)
            self.get_output(output_name).configure(dtype=self.input.dtype, shape=(self.nb_samples, shape))

        return

    def _process(self):

        batch = self.input.receive()
        for k in range(0, self.nb_groups):
            output_name = 'data_{}'.format(k)
            self.get_output(output_name).send(batch[:, k::self.nb_groups])

        return
