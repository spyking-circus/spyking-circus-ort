class Cell(object):
    # TODO add docstring.

    def __init__(self, template, train):

        self.template = template
        self.train = train

        # TODO correct the following lines.
        self.x = lambda _: 0.0
        self.y = lambda _: 0.0
        self.z = lambda _: 0.0
        self.e = lambda _, probe: 0
        self.r = lambda _: 0.0

        raise NotImplementedError()  # TODO complete.

    def get_waveform(self):
        # TODO add docstring.

        raise NotImplementedError()  # TODO complete.

    def get_waveforms(self):
        # TODO add docstring.

        raise NotImplementedError()  # TODO complete.

    def generate_spike_trains(self):
        # TODO add docstring.

        raise NotImplementedError()  # TODO complete.
