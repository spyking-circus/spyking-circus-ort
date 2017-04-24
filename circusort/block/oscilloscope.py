from .block import Block
import numpy
import time
import pylab
import os 

class Oscilloscope(Block):
    '''TODO add docstring'''

    name = "Oscilloscope"

    params = {'spacing' : 1, 
              'data_path' : None}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_input('data')
        self.add_input('peaks')
        self.add_input('mads')

    def _initialize(self):

        if self.data_path is None:
            self.data_path = self._get_tmp_path()
        
        self.data_path = os.path.abspath(os.path.expanduser(self.data_path))
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.log.info('Templates data are saved in {k}'.format(k=self.data_path))

        pylab.figure()
        return

    @property
    def nb_channels(self):
        return self.inputs['data'].shape[0]

    @property
    def nb_samples(self):
        return self.inputs['data'].shape[1]

    def _guess_output_endpoints(self):
        pass
    
    def _process(self):

        thresholds = self.inputs['mads'].receive(blocking=False)
        peaks = self.inputs['peaks'].receive(blocking=False)
        batch = self.inputs['data'].receive()
        pylab.gca().clear()

        for i in xrange(self.nb_channels):
            offset = self.spacing*i
            pylab.plot(offset + batch[i, :], '0.5')

            if thresholds is not None:
                pylab.plot([0, self.nb_samples], [offset - thresholds[i], offset + thresholds[i]], 'k--')

        if peaks is not None:
            tmp = peaks.pop('offset')
            print self.counter, tmp/self.nb_samples
            for key in peaks.keys():
                for channel in peaks[key].keys():
                    data = peaks[key][channel]
                    pylab.plot(data, self.spacing*int(channel)*numpy.ones(len(data)), 'r.')

        pylab.xlim(0, self.nb_samples)
        pylab.xlabel('Time [steps]')
        pylab.title('Buffer %d' %self.counter)

        pylab.savefig(os.path.join(self.data_path, 'oscillo_%d.png' %self.counter))

        return