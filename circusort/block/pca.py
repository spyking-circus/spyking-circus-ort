from .block import Block
import numpy as np
import scipy.interpolate

from circusort.utils.algorithms import PCAEstimator


class Pca(Block):
    """PCA

    Attributes:
        spike_width
        output_dim
        alignment
        nb_waveforms
        sampling_rate

    Inputs:
        data
        peaks

    Output:
        pcs

    """
    # TODO complete docstring.

    name = "PCA"

    params = {
        'spike_width': 5,
        'output_dim': 5,
        'alignment': True,
        'nb_waveforms': 10000,
        'sampling_rate': 20000,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        # The following lines are useful to avoid some PyCharm's warnings.
        self.spike_width = self.spike_width
        self.output_dim = self.output_dim
        self.alignment = self.alignment
        self.nb_waveforms = self.nb_waveforms
        self.sampling_rate = self.sampling_rate

        self.add_output('pcs')
        self.add_input('data')
        self.add_input('peaks')

    @property
    def nb_channels(self):
        return self.inputs['data'].shape[1]

    @property
    def nb_samples(self):
        return self.inputs['data'].shape[0]

    def _initialize(self):
        self._spike_width_ = int(self.sampling_rate*self.spike_width*1e-3)
        self.sign_peaks = None
        self.send_pcs = True
        if np.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width = (self._spike_width_-1)//2
        self._2_width = 2 * self._width

        if self.alignment:
            self.cdata = np.linspace(-self._width, self._width, 5*self._spike_width_)
            self.xdata = np.arange(-self._2_width, self._2_width + 1)
            self.xoff = len(self.cdata)/2.

        return

    def _guess_output_endpoints(self):
        self.outputs['pcs'].configure(dtype='float32', shape=(2, self._spike_width_, self.output_dim))
        self.pcs = np.zeros((2, self._spike_width_, self.output_dim), dtype=np.float32)

    def _is_valid(self, peak):
        if self.alignment:
            return (peak >= self._2_width) and (peak + self._2_width < self.nb_samples)
        else:
            return (peak >= self._width) and (peak + self._width < self.nb_samples)

    def is_ready(self, key=None):
        if key is not None:
            # # TODO remove the following lines.
            # print("nb_spikes: {}".format(self.nb_spikes[key]))
            # print("nb_waveforms: {}".format(self.nb_waveforms))
            # print("has_pcs: {}".format(self.has_pcs[key]))
            # TODO clean.
            # return (self.nb_spikes[key] == self.nb_waveforms) and not self.has_pcs[key]
            return (self.nb_spikes[key] >= self.nb_waveforms) and not self.has_pcs[key]
        else:
            # # TODO remove the following line.
            # print("has_pcs: {}".format(self.has_pcs))
            return bool(np.prod([i for i in self.has_pcs.values()]))  # TODO correct (use np.all instead)?

    def _get_waveform(self, batch, channel, peak, key):
        if self.alignment:
            ydata = batch[peak - self._2_width:peak + self._2_width + 1, channel]
            f = scipy.interpolate.UnivariateSpline(self.xdata, ydata, s=0)
            if key == 'negative':
                rmin = (np.argmin(f(self.cdata)) - self.xoff)/5.
            else:
                rmin = (np.argmax(f(self.cdata)) - self.xoff)/5.
            ddata = np.linspace(rmin - self._width, rmin + self._width, self._spike_width_)

            result = f(ddata).astype(np.float32)
        else:
            result = batch[peak - self._width:peak + self._width + 1, channel]
        return result

    def _infer_sign_peaks(self, peaks):
        self.sign_peaks = [str(i) for i in peaks.keys()]

        self.nb_spikes = {}
        self.waveforms = {}
        self.has_pcs = {}
        for key in peaks.keys():
            self.nb_spikes[key] = 0
            self.has_pcs[key] = False
            self.waveforms[key] = np.zeros((self.nb_waveforms, self._spike_width_), dtype=np.float32)

    def _process(self):

        batch = self.inputs['data'].receive()

        if self.is_active:
            peaks = self.inputs['peaks'].receive()
        else:
            peaks = self.inputs['peaks'].receive(blocking=False)

        if peaks is not None:

            self._measure_time('start', frequency=100)

            while not self._sync_buffer(peaks, self.nb_samples):
                peaks = self.inputs['peaks'].receive(blocking=True)

            _ = peaks.pop('offset')

            if self.sign_peaks is None:
                self._infer_sign_peaks(peaks)

            if not self.is_active:
                self._set_active_mode()

            if self.send_pcs:

                for key in self.sign_peaks:

                    for channel, signed_peaks in peaks[key].items():
                        if self.nb_spikes[key] < self.nb_waveforms:
                            for peak in signed_peaks:
                                if self.nb_spikes[key] < self.nb_waveforms and self._is_valid(peak):
                                    waveform = self._get_waveform(batch, int(channel), peak, key)
                                    self.waveforms[key][self.nb_spikes[key]] = waveform
                                    self.nb_spikes[key] += 1

                    if self.is_ready(key):
                        string = "{n} computes the PCA matrix from {k} {m} spikes"
                        message = string.format(n=self.name_and_counter, k=len(self.waveforms[key]), m=key)
                        self.log.info(message)
                        pca = PCAEstimator(self.output_dim)
                        pca.fit(self.waveforms[key])

                        np.save('pca_%s' %key, self.waveforms[key])

                        if key == 'negative':
                            self.pcs[0] = pca.components_.T
                        elif key == 'positive':
                            self.pcs[1] = pca.components_.T
                        self.has_pcs[key] = True

                if self.is_ready():
                    self.outputs['pcs'].send(self.pcs)
                    self.send_pcs = False

            self._measure_time('end', frequency=100)

        return

    def _introspect(self):
        # TODO add docstring.

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self.nb_samples) / self.sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return
