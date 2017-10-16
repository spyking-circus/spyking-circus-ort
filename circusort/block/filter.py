from .block import Block
import numpy
from scipy import signal

class Filter(Block):
    '''TODO add docstring'''

    name = "Filter"

    params = {'cut_off'       : 500,
              'sampling_rate' : 20000,
              'remove_median' : False}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('data')
        self.add_input('data')

    def _initialize(self):
        cut_off = numpy.array([self.cut_off, 0.95*(self.sampling_rate/2.)])
        b, a   = signal.butter(3, cut_off/(self.sampling_rate/2.), 'pass')
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
        # # TODO remove the following line.
        # self.log.debug("f>>>>>>>>>>")
        # # TODO remove the following 7 lines.
        # self.log.debug("f >>>>>>>>>>")
        # self.log.debug("f input addr: {}".format(self.input.addr))
        # self.log.debug("f input structure: {}".format(self.input.structure))
        # if self.input.structure == 'array':
        #     self.log.debug("f input dtype: {}".format(self.input.dtype))
        #     self.log.debug("f input shape: {}".format(self.input.shape))
        # self.log.debug("f output addr: {}".format(self.output.addr))
        batch = self.input.receive()
        # # TODO remove the following line.
        # self.log.debug("f ==========")
        try:
            # # TODO remove the following 2 lines.
            # if batch.dtype != numpy.float32:
            #     batch = batch.astype(numpy.float32)
            for i in xrange(self.nb_channels):
                batch[:, i], self.z[i]  = signal.lfilter(self.b, self.a, batch[:, i], zi=self.z[i])
                batch[:, i] -= numpy.median(batch[:, i])

            if self.remove_median:
                global_median = numpy.median(batch, 1)
                for i in xrange(self.nb_channels):
                    batch[:, i] -= global_median
            self.output.send(batch)
            # # TODO remove the following 2 lines.
            # self.log.debug("batch.shape: {}".format(batch.shape))
            # self.log.debug("batch.dtype: {}".format(batch.dtype))
            # # TODO remove the following line.
            # self.log.debug("f <<<<<<<<<<")
        except Exception as exception:
            self.log.debug("{} raised {}".format(self.name, exception))

        return