from .block import Block
import numpy


class Peak_detector(Block):
    '''TODO add docstring'''

    name = "Peak detector"

    params = {'sign_peaks'    : 'negative',
              'threshold'     : 6,
              'spike_width'   : 5,
              'sampling_rate' : 20000}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('peaks', 'dict')
        self.add_input('mads')
        self.add_input('data')

    def _initialize(self):
        self.peaks = {}
        if self.sign_peaks in ['negative', 'both']:
            self.peaks['negative'] = {}
        if self.sign_peaks in ['positive', 'both']:
            self.peaks['positive'] = {}

        self._spike_width_ = int(self.sampling_rate*self.spike_width*1e-3)
        self.sign_peaks    = None
        self.send_pcs      = True
        if numpy.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        return

    @property
    def nb_channels(self):
        return self.inputs['data'].shape[0]
        
    @property
    def nb_samples(self):
        return self.inputs['data'].shape[1]

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
            thresholds *= self.threshold
            for key in self.peaks.keys():
                self.peaks[key] = {}
                for i in xrange(self.nb_channels):
                    if key == 'negative':
                        self.peaks[key][i] = self._detect_peaks(batch[i],  thresholds[i], mpd=self._spike_width_)
                    elif key == 'positive':
                        self.peaks[key][i] = self._detect_peaks(batch[i],  thresholds[i], valley=True, mpd=self._spike_width_)

            self.outputs['peaks'].send(self.peaks)
        return