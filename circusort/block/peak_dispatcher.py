from .block import Block
import numpy

class Peak_dispatcher(Block):
    '''TODO add docstring'''

    name   = "Peak Dispatcher"

    params = {'nb_groups'  : 1,
              'nb_samples' : 1024}

    def __init__(self, **kwargs):
        Block.__init__(self, **kwargs)
        self.add_input('peaks')
        for i in xrange(self.nb_groups):
            self.add_output('peaks_%d' %i, 'dict')    
        
    def _initialize(self):
        self.ranges = [int(i*self.nb_samples/self.nb_groups) for i in xrange(self.nb_groups)]
        self.sign_peaks = None

    def _infer_sign_peaks(self, peaks):
        self.sign_peaks = [str(i) for i in peaks.keys()]

    def _init_data_structures(self):
        self.to_send = [{} for i in xrange(self.nb_groups)]
        
        if self.sign_peaks is None:
            self._infer_sign_peaks(peaks)

        for key in self.sign_peaks:
            for i in xrange(self.nb_groups):
                self.to_send[i][key] = {}


    def _guess_output_endpoints(self):
        pass

    def _process(self):
        peaks  = self.input.receive()
        offset = peaks.pop['offset']

        self._init_data_structures()

        for key in peaks.keys():
            for channel in peaks[key]:
                idx = numpy.searchsorted(peaks[key][channel], self.ranges)


        for i in xrange(self.nb_groups):
            self.get_output('peaks_%d' %i).send(self.to_send[i])
        return