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
        self.ranges = numpy.array([int(i*self.nb_samples/self.nb_groups) for i in xrange(self.nb_groups)], dtype=numpy.int32)
        self.sign_peaks = None

    def _infer_sign_peaks(self, peaks):
        self.sign_peaks = [str(i) for i in peaks.keys()]

    def _init_data_structures(self, offset=0):
        self.to_send = [{} for i in xrange(self.nb_groups)]
        self.offsets = offset + self.ranges

        for i in xrange(self.nb_groups):
            for key in self.sign_peaks:
                self.to_send[i][key] = {}
            self.to_send[i]['offset'] = int(self.offsets[i])

    def _guess_output_endpoints(self):
        pass

    def _process(self):
        peaks  = self.input.receive()
        offset = peaks.pop('offset')

        if self.sign_peaks is None:
            self._infer_sign_peaks(peaks)

        self._init_data_structures(offset)

        for key in peaks.keys():
            for channel in peaks[key]:
                idx = numpy.searchsorted(peaks[key][channel], self.ranges)
                idx = numpy.concatenate((idx, [peaks[key][channel][-1]]))
                for i in xrange(self.nb_groups):
                    if idx[i] != idx[i+1]:
                        self.to_send[i][key][channel] = peaks[key][channel][idx[i]:idx[i+1]] - self.ranges[i]

        for i in xrange(self.nb_groups):
            self.get_output('peaks_%d' %i).send(self.to_send[i])
        return