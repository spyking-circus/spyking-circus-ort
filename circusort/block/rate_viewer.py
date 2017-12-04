from .block import Block
from circusort.io.probe import load_probe
import numpy
import time
import pylab
import os


class Rate_viewer(Block):
    '''TODO add docstring'''

    name = "Rate viewer"

    params = {'probe'         : None,
              'sampling_rate' : 20000,
              'nb_samples'    : 1024}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        if self.probe == None:
            self.log.error('{n}: the probe file must be specified!'.format(n=self.name))
        else:
            self.probe = load_probe(self.probe, radius=None, logger=self.log)
            self.log.info('{n} reads the probe layout'.format(n=self.name))

        self.positions = self.probe.positions
        self.add_input('peaks')
        self.mpl_display = True

    def _initialize(self):

        self.data_available = False
        self.batch = None
        self.thresholds = None
        self.peaks = None
        self.data_lines = None
        self.threshold_lines = None
        self.peak_points = None
        self.sign_peaks = None
        self.rates = {}
        return

    def _infer_sign_peaks(self, peaks):
        self.sign_peaks = [str(i) for i in peaks.keys()]

    def _guess_output_endpoints(self):
        pass

    def _process(self):

        peaks      = self.inputs['peaks'].receive(blocking=False)
        if peaks is not None:

            if not self.is_active:
                self._set_active_mode()

            peaks.pop('offset')
            #while peaks.pop('offset')/self.nb_samples < self.counter:
            #    peaks = self.inputs['peaks'].receive()

            if self.sign_peaks is None:
                self._infer_sign_peaks(peaks)

            if self.rates == {}:
                for key in self.sign_peaks:
                    self.rates[key] = numpy.zeros(self.probe.nb_channels)

        self.peaks = peaks
        self.data_available = True

    def _plot(self):
        # Called from the main thread
        pylab.ion()

        if not getattr(self, 'data_available', False):
            return

        if self.peaks is not None:
            
            for key in self.sign_peaks:
                for channel in self.peaks[key].keys():
                    self.rates[key][int(channel)] += len(self.peaks[key][channel])

            pylab.scatter(self.positions[0, :], self.positions[1, :], c=self.rates[key])
            
        pylab.gca().set_title('Buffer %d' %self.counter)
        pylab.draw()
        return