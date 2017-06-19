from .block import Block
import numpy


class Peak_detector(Block):
    '''TODO add docstring'''

    name = "Peak detector"

    params = {'sign_peaks'    : 'negative',
              'spike_width'   : 5,
              'sampling_rate' : 20000.,
              'safety_time'   : 'auto'}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('peaks', 'dict')
        self.add_input('mads')
        self.add_input('data')

    def _initialize(self):
        self.peaks = {'offset' : 0}
        if self.sign_peaks == 'both':
            self.key_peaks = ['negative', 'positive']
        else:
            self.key_peaks = [self.sign_peaks]
        self._spike_width_ = int(self.sampling_rate*self.spike_width*1e-3)
        if numpy.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width = self._spike_width_/2
        is self.safety_time == 'auto':
            self.safety_time = self._width
        else:
            self.safety_time = int(self.sampling_rate*self.safety_time*1e-3)
        return

    @property
    def nb_channels(self):
        return self.inputs['data'].shape[1]
        
    @property
    def nb_samples(self):
        return self.inputs['data'].shape[0]

    def _guess_output_endpoints(self):
        return

    def _detect_peaks(self, x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

        if valley:
            x = -x
        # find indices of all peaks
        dx = x[1:] - x[:-1]
        ine, ire, ife = numpy.array([[], [], []], dtype=numpy.int32)
        if not edge:
            ine = numpy.where((numpy.hstack((dx, 0)) < 0) & (numpy.hstack((0, dx)) > 0))[0]
        else:
            if edge.lower() in ['rising', 'both']:
                ire = numpy.where((numpy.hstack((dx, 0)) <= 0) & (numpy.hstack((0, dx)) > 0))[0]
            if edge.lower() in ['falling', 'both']:
                ife = numpy.where((numpy.hstack((dx, 0)) < 0) & (numpy.hstack((0, dx)) >= 0))[0]
        ind = numpy.unique(numpy.hstack((ine, ire, ife)))
        # first and last values of x cannot be peaks
        if ind.size and ind[0] == 0:
            ind = ind[1:]
        if ind.size and ind[-1] == x.size-1:
            ind = ind[:-1]
        # remove peaks < minimum peak height
        if ind.size and mph is not None:
            ind = ind[x[ind] >= mph]
        # remove peaks - neighbors < threshold
        if ind.size and threshold > 0:
            dx = numpy.min(numpy.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
            ind = numpy.delete(ind, numpy.where(dx < threshold)[0])
        # detect small peaks closer than minimum peak distance
        if ind.size and mpd > 1:
            ind = ind[numpy.argsort(x[ind])][::-1]  # sort ind by peak height
            idel = numpy.zeros(ind.size, dtype=numpy.bool)
            for i in range(ind.size):
                if not idel[i]:
                    # keep peaks with the same height if kpsh is True
                    idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                        & (x[ind[i]] > x[ind] if kpsh else True)
                    idel[i] = 0  # Keep current peak
            # remove the small peaks and sort back the indices by their occurrence
            ind = numpy.sort(ind[~idel])

        return ind

    def _process(self):
        batch      = self.get_input('data').receive()
        thresholds = self.get_input('mads').receive(blocking=False)

        if thresholds is not None:

            if not self.is_active:
                self._set_active_mode()

            for key in self.key_peaks:
                self.peaks[key] = {}
                for i in xrange(self.nb_channels):
                    if key == 'negative':
                        data = self._detect_peaks(batch[:, i],  thresholds[i], valley=True, mpd=self.safety_time)
                        if len(data) > 0:
                            self.peaks[key][i] = data
                    elif key == 'positive':
                        data = self._detect_peaks(batch[:, i],  thresholds[i], valley=False, mpd=self.safety_time)
                        if len(data) > 0:
                            self.peaks[key][i] = data

            self.peaks['offset'] = self.counter*self.nb_samples
            self.outputs['peaks'].send(self.peaks)
        return