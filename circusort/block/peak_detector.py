# import matplotlib.pyplot as plt
import numpy

from .block import Block


class Peak_detector(Block):
    """TODO add docstring"""

    name = "Peak detector"

    params = {
        'sign_peaks': 'negative',
        'spike_width': 5,
        'sampling_rate': 20000.,
        'safety_time': 'auto',
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('peaks', 'dict')
        self.add_input('mads')
        self.add_input('data')

    def _initialize(self):

        self.peaks = {
            'offset': 0,
            'negative': {},
            'positive': {},
        }
        if self.sign_peaks == 'both':
            self.key_peaks = ['negative', 'positive']
        else:
            self.key_peaks = [self.sign_peaks]
        self._spike_width_ = int(self.sampling_rate*self.spike_width*1e-3)
        if numpy.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width = self._spike_width_/2
        if self.safety_time == 'auto':
            self.safety_time = self._width
        else:
            self.safety_time = max(1, int(self.sampling_rate * self.safety_time * 1e-3))

        # Internal variables for peak detection.
        self.X = None
        self.e = None
        self.p = None
        self.mph = None

        return

    @property
    def nb_channels(self):
        return self.inputs['data'].shape[1]

    @property
    def nb_samples(self):
        return self.inputs['data'].shape[0]

    def _guess_output_endpoints(self):
        return

    def _detect_peaks(self, i, mpd=1, threshold=0.0, edge='rising', kpsh=False, valley=False):
        """Detect peaks

        Parameters
        ----------
        i: integer
            Channel identifier.
        mpd: integer, optional
            Minimum peak distance. The default value is 1.
        threshold: float, optional
            Minimum threshold between the peak and its two neighboring values. The default value is 0.0.
        edge: None | 'rising' | 'falling' | 'both', optional
            Type of edges to detect during the peak detection. the default value is 'rising'.
        kpsh: boolean
            Option to keep peaks with same height. The default value is False.
        valley: boolean
            Option to detect valleys instead of peaks. The default value is False.

        """

        if valley:
            x = -self.X[:, i]
        else:
            x = +self.X[:, i]

        # Find indices of all edges.
        dx = x[self.nb_samples-1:] - x[self.nb_samples-2:-1]
        ne = numpy.zeros(self.nb_samples, dtype=numpy.bool)
        re = numpy.zeros(self.nb_samples, dtype=numpy.bool)
        fe = numpy.zeros(self.nb_samples, dtype=numpy.bool)
        if not edge:
            ne = (dx[+1:] < 0.0) & (dx[:-1] > 0.0)
        else:
            if edge.lower() in ['rising', 'both']:
                re = (dx[+1:] <= 0.0) & (dx[:-1] > 0.0)
            if edge.lower() in ['rising', 'both']:
                fe = (dx[+1:] < 0.0) & (dx[:-1] >= 0.0)
        e = numpy.logical_or(ne, numpy.logical_or(re, fe))
        self.e[self.nb_samples-1:2*self.nb_samples-1, i] = e
        ind = numpy.add(numpy.where(e)[0], self.nb_samples - 1)
        # Remove edges < minimum peak height.
        if self.mph is not None:
            self.e[ind, i] = (x[ind] >= self.mph[i])
            ind = ind[self.e[ind, i]]
        # Remove peak - neighbors < threshold
        if threshold > 0.0:
            dx = numpy.min(numpy.vstack((x[ind] - x[ind-1], x[ind] - x[ind+1])))
            self.e[ind, i] = (dx < threshold)
            # TODO remove the following line.
            # ind = ind[self.e[ind, i]]
        # Detect small edges closer than minimum peak distance.
        # TODO remove the three following lines.
        # e = self.e[self.nb_samples-mpd-1:2*self.nb_samples-mpd-1, i]
        # ind = numpy.add(numpy.where(e)[0], self.nb_samples - mpd - 1)
        # self.p[ind, i] = True
        p = self.e[self.nb_samples-2*mpd-1:2*self.nb_samples-1, i]
        ind = numpy.where(p)[0]
        ind_ = numpy.add(ind, self.nb_samples - 2 * mpd - 1)
        if mpd > 1:
            ind = ind[numpy.argsort(x[ind_])][::-1]
            for j in range(0, ind.size):
                if p[ind[j]]:
                    # Keep peaks with the same height if 'kpsh' is True.
                    i_del = (ind >= ind[j] - mpd) & (ind <= ind[j] + mpd) & (x[ind_[j]] > x[ind_] if kpsh else True)
                    p[ind[i_del]] = False
                    # Keep current peak.
                    p[ind[j]] = True
                else:
                    pass
        self.p[self.nb_samples-mpd-1:2*self.nb_samples-mpd-1, i] = p[+mpd:-mpd]

        # Return detected peaks from the previous chunk of data.
        if self.counter <= self.start_step + 1:
            # TODO correct the following block code (doesn't work).
            # We need to discard the beginning of the first chunk (i.e. where we can't define any peak).
            p = self.p[mpd+1:self.nb_samples, i]
            ind = numpy.add(numpy.where(p)[0], mpd + 1)
        else:
            p = self.p[0:self.nb_samples, i]
            ind = numpy.add(numpy.where(p)[0], 0)

        return ind

    def _process(self):
        """Process data streams"""

        # Update internal variables for peak detection.
        if self.counter == 0:
            self.X = numpy.zeros((2 * self.nb_samples, self.nb_channels), dtype=numpy.float)
            self.X[self.nb_samples:, :] = self.get_input('data').receive()
            self.X[:self.nb_samples, :] =\
                numpy.repeat(self.X[self.nb_samples, :], self.nb_samples).reshape((self.nb_samples, self.nb_channels))
            self.e = numpy.zeros((2 * self.nb_samples, self.nb_channels), dtype=numpy.bool)
            self.e[:self.nb_samples, :] = self.e[self.nb_samples:, :]
            self.e[self.nb_samples:, :] = numpy.zeros((self.nb_samples, self.nb_channels), dtype=numpy.bool)
            self.p = numpy.zeros((2 * self.nb_samples, self.nb_channels), dtype=numpy.bool)
            self.p[:self.nb_samples, :] = self.p[self.nb_samples:, :]
            self.p[self.nb_samples:, :] = numpy.zeros((self.nb_samples, self.nb_channels), dtype=numpy.bool)
            self.mph = numpy.zeros((self.nb_channels,), dtype=numpy.float)
            self.mph = self.get_input('mads').receive(blocking=False)
        else:
            self.X[:self.nb_samples, :] = self.X[self.nb_samples:, :]
            self.X[self.nb_samples:, :] = self.get_input('data').receive()
            self.e[:self.nb_samples, :] = self.e[self.nb_samples:, :]
            self.e[self.nb_samples:, :] = numpy.zeros((self.nb_samples, self.nb_channels), dtype=numpy.bool)
            self.p[:self.nb_samples, :] = self.p[self.nb_samples:, :]
            self.p[self.nb_samples:, :] = numpy.zeros((self.nb_samples, self.nb_channels), dtype=numpy.bool)
            self.mph = self.get_input('mads').receive(blocking=False)

        # If median absolute deviations are defined...
        if self.mph is not None:

            if not self.is_active:
                self._set_active_mode()

            for key in self.key_peaks:
                self.peaks[key] = {}
                for i in range(0, self.nb_channels):
                    if key == 'negative':
                        data = self._detect_peaks(i, valley=True, mpd=self.safety_time)
                        if len(data) > 0:
                            self.peaks[key][i] = data
                    elif key == 'positive':
                        data = self._detect_peaks(i, valley=False, mpd=self.safety_time)
                        if len(data) > 0:
                            self.peaks[key][i] = data
            # TODO check the following correction.
            # self.peaks['offset'] = self.counter * self.nb_samples
            self.peaks['offset'] = (self.counter - 1) * self.nb_samples

            # Send detected peaks.
            self.outputs['peaks'].send(self.peaks)

        else:

            if self.is_active:
                raise Exception("Peak detector is active but receive no MADs (counter={})".format(self.counter))

        return
