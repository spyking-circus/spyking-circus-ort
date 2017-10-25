from .block import Block
import numpy
from scipy import signal


class Filter(Block):
    """Filtering of the voltage traces of the recording channels

    Parameters
    ----------
    sampling_rate: int (default 20000)
        Sampling rate (in Hz) used to record the signal.
    cut_off: int (default 500)
        Cutoff frequency (in Hz) used to define the high-pass filter
        (third-order Butterworth).
    remove_median: bool (default False)
        Option to remove the median over all the channels for each time step.

    See Also
    --------
    Block
    """
    # TODO complete docstring.

    name = "Filter"

    params = {
        'cut_off': 500,  # Hz
        'sampling_rate': 20000,  # Hz
        'remove_median': False,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('data')
        self.add_input('data')

    def _initialize(self):
        cut_off = numpy.array([self.cut_off, 0.95*(self.sampling_rate/2.)])
        b, a = signal.butter(3, cut_off/(self.sampling_rate/2.), 'pass')
        self.b = b
        self.a = a
        self.z = {}
        return

    @property
    def nb_channels(self):
        return self.input.shape[1]

    @property
    def nb_samples(self):
        return self.input.shape[0]

    def _guess_output_endpoints(self):
        self.output.configure(dtype=self.input.dtype, shape=self.input.shape)
        self.z = {}
        m = max(len(self.a), len(self.b)) - 1
        for i in xrange(self.nb_channels):
            self.z[i] = numpy.zeros(m, dtype=numpy.float32)

    def _process(self):

        # Receive input data.
        batch = self.input.receive()
        # Process data.
        for i in xrange(self.nb_channels):
            batch[:, i], self.z[i] = signal.lfilter(self.b, self.a, batch[:, i], zi=self.z[i])
            batch[:, i] -= numpy.median(batch[:, i])
        if self.remove_median:
            global_median = numpy.median(batch, 1)
            for i in xrange(self.nb_channels):
                batch[:, i] -= global_median
        # Send output data.
        self.output.send(batch)

        return
