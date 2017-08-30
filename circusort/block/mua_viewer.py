import matplotlib.pyplot as plt
import numpy as np

from .block import Block
from circusort.io.probe import Probe


class Mua_viewer(Block):
    """Multi-unit activity viewer

    probe: string
        Name of the probe.
    sampling_rate: float, optional
        Sampling rate [Hz]. The default value is 20.0e+3
    nb_samples: integer, optional
        Number of samples for each chunk of data. The default value is 1024.
    time_window: float, optional
        Size of the time window used to define the multi-unit activity [ms]. The default value is 50.0.

    See also circusort.block.Block.

    """

    name = "MUA viewer"

    params = {
        'probe': None,
        'sampling_rate': 20.0e+3,  # Hz
        'nb_samples': 1024,
        'time_window': 50.0,  # ms
        'c_max': 5.0,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        if self.probe is None:
            # TODO improve the following line.
            raise NotImplementedError()
        else:
            self.probe = Probe(self.probe, radius=None, logger=self.log)

        self.add_input('peaks')

        # Flag to call plot from the main thread (c.f. circusort.cli.process).
        self.mpl_display = True

    def _initialize(self):
        """Initialize this block"""

        self.data_available = False
        self.peaks = None
        self.peak_points = None

        return

    @property
    def nb_channels(self):
        """Number of channels"""
        return self.probe.nb_channels

    def _process(self):
        """Process next buffer"""

        peaks = self.inputs['peaks'].receive(blocking=False)
        if peaks is not None:
            if not self.is_active:
                self._set_active_mode()
            peaks.pop('offset')
            self.peaks = peaks

        self.data_available = True

        return

    def _plot(self):
        """Plot viewer"""

        # Called from the main thread.
        plt.ion()

        if not getattr(self, 'data_available', False):
            return

        self.data_available = False

        # Plot detected peaks.
        if self.peaks is not None:
            peaks_list = [(self.peaks[key][channel], channel) for key in self.peaks for channel in self.peaks[key]]
            if peaks_list:
                data, channel = zip(*peaks_list)
                lengths = [len(d) for d in data]
                channel = np.repeat(np.int_(channel), lengths)
                data = np.hstack(data)
            else:
                channel = np.array([], dtype=np.int)
                data = np.array([], dtype=np.int)
            if self.peak_points is None:
                self.peak_points = plt.scatter(data, channel, color='C0')
            else:
                offsets = np.transpose(np.stack((data, channel)))
                self.peak_points.set_offsets(offsets)

        ax = plt.gca()
        # Set limits.
        ax.set_xlim(float(0) - 0.5, float(self.nb_samples - 1) + 0.5)
        ax.set_ylim(float(0) - 0.5, float(self.nb_channels - 1) + 0.5)
        # Set labels.
        ax.set_xlabel("time (arb.unit)")
        ax.set_ylabel("channel")
        # Set title.
        ax.set_title("Buffer {}".format(self.counter))

        # Draw viewer.
        plt.draw()

        return
