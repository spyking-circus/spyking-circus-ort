import numpy as np

from .block import Block


__classname__ = "PeakDetector"


class PeakDetector(Block):
    """Peak detector block

    Attributes:
        threshold_factor
        sign_peaks
        spike_width
        sampling_rate
        safety_time

    Inputs:
        data: np.ndarray
            Voltage signal.
        mads

    Outputs:
        peaks: dictionary
            'offset' -> integer
                Chunk offset.
            'positive' -> np.ndarray (optional)
                Time step of each positive peak detected in the chunk.
            'negative' -> np.ndarray (optional)
                Time step of each negative peak detected in the chunk.
        dict
    """

    name = "Peak detector"

    params = {
        'threshold_factor': 5.0,
        'sign_peaks': 'negative',
        'spike_width': 5.0,  # ms
        'sampling_rate': 20e+3,  # Hz
        'safety_time': 'auto',
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('peaks', structure='dict')
        self.add_input('mads', structure='dict')
        self.add_input('data', structure='dict')

        # The following lines are useful to avoid some PyCharm warnings.
        self.threshold_factor = self.threshold_factor
        self.sign_peaks = self.sign_peaks
        self.spike_width = self.spike_width
        self.sampling_rate = self.sampling_rate
        self.safety_time = self.safety_time

        self._nb_channels = None
        self._nb_samples = None

    def _initialize(self):

        self.peaks = {'offset': 0}
        if self.sign_peaks == 'both':
            self.key_peaks = ['negative', 'positive']
        else:
            self.key_peaks = [self.sign_peaks]
        self.nb_cum_peaks = {key: {} for key in self.key_peaks}
        self._spike_width_ = int(self.sampling_rate * self.spike_width * 1e-3)
        if np.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width = self._spike_width_ // 2
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

    def _configure_input_parameters(self, nb_channels=None, nb_samples=None, **kwargs):

        if nb_channels is not None:
            self._nb_channels = nb_channels
        if nb_samples is not None:
            self._nb_samples = nb_samples

        return

    def _update_initialization(self):

        shape = (2 * self._nb_samples, self._nb_channels)

        self.X = np.zeros(shape, dtype=np.float)
        self.e = np.zeros(shape, dtype=np.bool)
        self.p = np.zeros(shape, dtype=np.bool)
        self.mph = None

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
        dx = x[self._nb_samples-1:] - x[self._nb_samples-2:-1]
        ne = np.zeros(self._nb_samples, dtype=np.bool)
        re = np.zeros(self._nb_samples, dtype=np.bool)
        fe = np.zeros(self._nb_samples, dtype=np.bool)
        if not edge:
            ne = (dx[+1:] < 0.0) & (dx[:-1] > 0.0)
        else:
            if edge.lower() in ['rising', 'both']:
                re = (dx[+1:] <= 0.0) & (dx[:-1] > 0.0)
            if edge.lower() in ['rising', 'both']:
                fe = (dx[+1:] < 0.0) & (dx[:-1] >= 0.0)
        e = np.logical_or(ne, np.logical_or(re, fe))
        self.e[self._nb_samples-1:2*self._nb_samples-1, i] = e
        ind = np.add(np.where(e)[0], self._nb_samples - 1)
        # Remove edges < minimum peak height.
        if self.mph is not None:
            self.e[ind, i] = (x[ind] >= self.threshold_factor * self.mph[0, i])
            ind = ind[self.e[ind, i]]
        # Remove peak - neighbors < threshold
        if threshold > 0.0:
            dx = np.min(np.vstack((x[ind] - x[ind-1], x[ind] - x[ind+1])))
            self.e[ind, i] = (dx < threshold)
            # TODO remove the following line.
            # ind = ind[self.e[ind, i]]
        # Detect small edges closer than minimum peak distance.
        # TODO remove the three following lines.
        # e = self.e[self._nb_samples-mpd-1:2*self._nb_samples-mpd-1, i]
        # ind = np.add(np.where(e)[0], self._nb_samples - mpd - 1)
        # self.p[ind, i] = True
        p = self.e[self._nb_samples-2*mpd-1:2*self._nb_samples-1, i]
        ind = np.where(p)[0]
        ind_ = np.add(ind, self._nb_samples - 2 * mpd - 1)
        if mpd > 1:
            ind = ind[np.argsort(x[ind_])][::-1]
            for j in range(0, ind.size):
                if p[ind[j]]:
                    # Keep peaks with the same height if 'kpsh' is True.
                    i_del = (ind >= ind[j] - mpd) & (ind <= ind[j] + mpd) & (x[ind_[j]] > x[ind_] if kpsh else True)
                    p[ind[i_del]] = False
                    # Keep current peak.
                    p[ind[j]] = True
                else:
                    pass
        self.p[self._nb_samples-mpd-1:2*self._nb_samples-mpd-1, i] = p[+mpd:-mpd]

        # Return detected peaks from the previous chunk of data.
        if self.counter <= self.start_step + 1:
            # TODO correct the following block code (doesn't work).
            # We need to discard the beginning of the first chunk (i.e. where we can't define any peak).
            p = self.p[mpd+1:self._nb_samples, i]
            ind = np.add(np.where(p)[0], mpd + 1)
        else:
            p = self.p[0:self._nb_samples, i]
            ind = np.add(np.where(p)[0], 0)

        return ind

    def _process(self):
        """Process data streams"""

        # Update internal variable X.
        data_packet = self.get_input('data').receive()
        number = data_packet['number']
        data = data_packet['payload']
        if self.counter == 0:
            self.X[:self._nb_samples, :] = np.tile(data[0, :], (self._nb_samples, 1))
            self.X[self._nb_samples:, :] = data
        else:
            self.X[:self._nb_samples, :] = self.X[self._nb_samples:, :]
            self.X[self._nb_samples:, :] = data
        # Update internal variable e.
        self.e[:self._nb_samples, :] = self.e[self._nb_samples:, :]
        self.e[self._nb_samples:, :] = np.zeros((self._nb_samples, self._nb_channels), dtype=np.bool)
        # Update internal variable p.
        self.p[:self._nb_samples, :] = self.p[self._nb_samples:, :]
        self.p[self._nb_samples:, :] = np.zeros((self._nb_samples, self._nb_channels), dtype=np.bool)
        # Update internal variable mph.
        # TODO swap and clean the 2 following lines.
        # mph_packet = self.get_input('mads').receive(blocking=False, number=number)
        mph_packet = self.get_input('mads').receive(blocking=False)
        self.mph = mph_packet['payload'] if mph_packet is not None else self.mph

        self._measure_time('start')

        # If median absolute deviations are defined...
        if self.mph is not None:

            if not self.is_active:
                self._set_active_mode()

            for key in self.key_peaks:
                self.peaks[key] = {}
                for i in range(0, self._nb_channels):
                    if i not in self.nb_cum_peaks[key]:
                        self.nb_cum_peaks[key][i] = 0
                    if key == 'negative':
                        data = self._detect_peaks(i, valley=True, mpd=self.safety_time)
                        if len(data) > 0:
                            self.peaks[key][i] = data
                            self.nb_cum_peaks[key][i] += len(data)
                    elif key == 'positive':
                        data = self._detect_peaks(i, valley=False, mpd=self.safety_time)
                        if len(data) > 0:
                            self.peaks[key][i] = data
                            self.nb_cum_peaks[key][i] += len(data)

            self.peaks['offset'] = (self.counter - 1) * self._nb_samples

            # Prepare output packet.
            packet = {
                'number': number - 1,
                'payload': self.peaks,
            }

            # Send detected peaks.
            self.get_output('peaks').send(packet)

        self._measure_time('end')

        return

    def _introspect(self):

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self._nb_samples) / self.sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        # Log info message.
        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return
