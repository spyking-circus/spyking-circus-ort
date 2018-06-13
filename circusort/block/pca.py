from .block import Block
import numpy as np
import scipy.interpolate
from sklearn.decomposition import PCA


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

        self.add_output('pcs', structure='dict')
        self.add_input('data', structure='dict')
        self.add_input('peaks', structure='dict')

        self._nb_channels = None
        self._nb_samples = None
        self._output_dtype = 'float32'
        self._output_shape = None
        self.pcs = None

    def _initialize(self):

        self._spike_width_ = int(self.sampling_rate * self.spike_width * 1e-3)
        self.sign_peaks = None
        self.send_pcs = True
        if np.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width = (self._spike_width_ - 1) // 2
        self._2_width = 2 * self._width

        if self.alignment:
            self.cdata = np.linspace(-self._width, self._width, 5 * self._spike_width_)
            self.xdata = np.arange(-self._2_width, self._2_width + 1)
            self.xoff = len(self.cdata) / 2.0

        self._output_shape = (2, self._spike_width_, self.output_dim)

        self.pcs = np.zeros(self._output_shape, dtype=self._output_dtype)

        return

    def _configure_input_parameters(self, nb_channels=None, nb_samples=None, **kwargs):

        if nb_channels is not None:
            self._nb_channels = nb_channels
        if nb_samples is not None:
            self._nb_samples = nb_samples

        return

    def _is_valid(self, peak):

        if self.alignment:
            return (peak >= self._2_width) and (peak + self._2_width < self._nb_samples)
        else:
            return (peak >= self._width) and (peak + self._width < self._nb_samples)

    def is_ready(self, key=None):

        if key is not None:
            return (self.nb_spikes[key] >= self.nb_waveforms) and not self.has_pcs[key]
        else:
            return bool(np.prod([i for i in self.has_pcs.values()]))  # TODO correct (use np.all instead)?

    def _get_waveform(self, batch, channel, peak, key):

        if self.alignment:
            ydata = batch[peak - self._2_width:peak + self._2_width + 1, channel]
            f = scipy.interpolate.UnivariateSpline(self.xdata, ydata, s=0)
            if key == 'negative':
                rmin = float(np.argmin(f(self.cdata)) - self.xoff) / 5.0
            else:
                rmin = float(np.argmax(f(self.cdata)) - self.xoff) / 5.0
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

        return

    def _process(self):

        # Receive input data.
        data_packet = self.get_input('data').receive()
        number = data_packet['number']
        batch = data_packet['payload']
        # Receive peaks (if necessary).
        if self.is_active:
            peaks_packet = self.get_input('peaks').receive(blocking=True, number=number)
        else:
            peaks_packet = self.get_input('peaks').receive(blocking=False, number=number)
        peaks = peaks_packet['payload'] if peaks_packet is not None else None

        if peaks is not None:

            self._measure_time('start', frequency=100)

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
                        # Log info message.
                        string = "{} computes the PCA matrix from {} {} spikes"
                        message = string.format(self.name_and_counter, len(self.waveforms[key]), key)
                        self.log.info(message)
                        # Initialize and fit PCA.
                        pca = PCA(self.output_dim)
                        pca.fit(self.waveforms[key])

                        if key == 'negative':
                            self.pcs[0] = pca.components_.T
                        elif key == 'positive':
                            self.pcs[1] = pca.components_.T
                        self.has_pcs[key] = True

                if self.is_ready():
                    # Prepare output packet.
                    packet = {
                        'number': data_packet['number'],
                        'payload': self.pcs,
                    }
                    # Send principal components.
                    self.get_output('pcs').send(packet)
                    # Update internal variable.
                    self.send_pcs = False

            self._measure_time('end', frequency=100)

        return

    def _introspect(self):
        # TODO add docstring.

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
