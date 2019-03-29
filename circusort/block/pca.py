# -*- coding: utf-8 -*-
from .block import Block

# import matplotlib.pyplot as plt
import numpy as np
# import os

from sklearn.decomposition import PCA as PCA_

from circusort.obj.buffer import Buffer


__classname__ = 'PCA'


class PCA(Block):
    """PCA

    Attributes:
        spike_width: float
        spike_jitter: float
        spike_sigma: float
        output_dim: integer
        alignment: boolean
        nb_waveforms: integer
        sampling_rate: float

    Inputs:
        data
        peaks

    Output:
        pcs

    """

    name = "PCA"

    params = {
        'spike_width': 5.0,  # ms
        'spike_jitter': 1.0,  # ms
        'spike_sigma': 0.0,  # µV
        'output_dim': 5,
        'alignment': True,
        'nb_waveforms': 10000,
        'sampling_rate': 20e+3,  # Hz
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        # The following lines are useful to avoid some PyCharm's warnings.
        self.spike_width = self.spike_width
        self.spike_jitter = self.spike_jitter
        self.spike_sigma = self.spike_sigma
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

        self.sign_peaks = None
        self.send_pcs = True
        self.batch = Buffer(self.sampling_rate, self.spike_width, self.spike_jitter, alignment=self.alignment)
        self._output_shape = (2, self.batch.temporal_width, self.output_dim)
        self.pcs = np.zeros(self._output_shape, dtype=self._output_dtype)

        return

    def _configure_input_parameters(self, nb_channels=None, nb_samples=None, **kwargs):

        if nb_channels is not None:
            self._nb_channels = nb_channels
        if nb_samples is not None:
            self._nb_samples = nb_samples

        return

    def is_ready(self, key=None):

        if key is not None:
            return (self.nb_spikes[key] >= self.nb_waveforms) and not self.has_pcs[key]
        else:
            return bool(np.prod([i for i in self.has_pcs.values()]))  # TODO correct (use np.all instead)?

    def _infer_sign_peaks(self, peaks):

        self.sign_peaks = [str(i) for i in peaks.keys()]

        self.nb_spikes = {}
        self.waveforms = {}
        self.has_pcs = {}
        for key in peaks.keys():
            self.nb_spikes[key] = 0
            self.has_pcs[key] = False
            self.waveforms[key] = np.zeros((self.nb_waveforms, self.batch.temporal_width), dtype=np.float32)

        return

    def _process(self):

        # Receive input data.
        data_packet = self.get_input('data').receive()
        data = data_packet['payload']
        number = data_packet['number']
        offset = number * self._nb_samples
        self.batch.update(data, offset=offset)
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
                            # Collect more waveforms.
                            for peak in signed_peaks:
                                if self.nb_spikes[key] < self.nb_waveforms and self.batch.valid_peaks(peak):
                                    waveform = self.batch.get_waveform(int(channel), peak, peak_type=key,
                                                                       sigma=self.spike_sigma)
                                    self.waveforms[key][self.nb_spikes[key]] = waveform
                                    self.nb_spikes[key] += 1
                            # Log debug message (if necessary).
                            if self.counter % 50 == 0:
                                string = "{} gathers {} {} peaks ({} wanted)"
                                message = string.format(self.name_and_counter, self.nb_spikes[key], key,
                                                        self.nb_waveforms)
                                self.log.debug(message)

                    if self.is_ready(key):
                        # Log info message.
                        string = "{} computes the PCA matrix from {} {} spikes"
                        message = string.format(self.name_and_counter, len(self.waveforms[key]), key)
                        self.log.info(message)
                        # Initialize and fit PCA.
                        pca = PCA_(self.output_dim)
                        pca.fit(self.waveforms[key])

                        # TODO remove the following lines.
                        # # 1st plot.
                        # if not os.path.isdir("/tmp/waveforms"):
                        #     os.makedirs("/tmp/waveforms")
                        # for k in range(0, min(10, self.waveforms[key].shape[0])):
                        #     waveform = self.waveforms[key][k]
                        #     fig, ax = plt.subplots()
                        #     ax.plot(waveform, color='C0')
                        #     ax.axvline(x=(len(waveform) - 1) // 2, color='grey')
                        #     fig.savefig("/tmp/waveforms/{}_waveform_{}.pdf".format(key, k))
                        #     plt.close(fig)
                        # # 2nd plot.
                        # fig, ax = plt.subplots()
                        # waveform = None
                        # for k in range(0, min(1000, self.waveforms[key].shape[0])):
                        #     waveform = self.waveforms[key][k]
                        #     ax.plot(waveform, color='C0', linewidth=1, alpha=0.25)
                        # if waveform is not None:
                        #     ax.axvline(x=(len(waveform) - 1) // 2, color='grey')
                        # fig.savefig("/tmp/waveforms/{}_waveforms.pdf".format(key))
                        # plt.close(fig)
                        # # 3rd plot.
                        # fig, ax = plt.subplots()
                        # waveform = None
                        # for k in range(0, min(1000, self.waveforms[key].shape[0])):
                        #     waveform = self.waveforms[key][k]
                        #     central_time_step = (len(waveform) - 1) // 2
                        #     if np.argmin(waveform) != central_time_step:
                        #         ax.plot(waveform, color='C0', linewidth=1, alpha=0.25)
                        # if waveform is not None:
                        #     ax.axvline(x=(len(waveform) - 1) // 2, color='grey')
                        # fig.savefig("/tmp/waveforms/misaligned_{}_waveforms.pdf".format(key))
                        # plt.close(fig)

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
