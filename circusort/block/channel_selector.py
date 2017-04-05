from .block import Block


class Channel_selector(Block):
    '''TODO add docstring'''

    name   = "Channel Selector"

    params = {'channels'   : []}

    def __init__(self, **kwargs):
        Block.__init__(self, **kwargs)
        self.add_output('data')
        self.add_input('data')

        
    @property
    def nb_channels(self):
        if len(self.channels) > 0:
            nb_channels = len(self.channels)
        else:
            nb_channels = self.input.shape[0]
        return nb_channels

    @property
    def nb_samples(self):
        return self.input.shape[1]

    def _initialize(self):
        return

    def _guess_output_endpoints(self):
        self.output.configure(dtype=self.input.dtype, shape=(self.nb_channels, self.nb_samples))

    def _process(self):
        batch = self.input.receive()
        if len(self.channels) > 0:
            batch = batch[self.channels]

        self.output.send(batch.flatten())
        return