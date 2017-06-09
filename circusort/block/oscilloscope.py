from .block import Block
import numpy
import time
import pylab
import os


class Oscilloscope(Block):
    '''TODO add docstring'''

    name = "Oscilloscope"

    params = {'spacing'   : 1}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_input('data')
        self.add_input('peaks')
        self.add_input('mads')
        self.mpl_display = True

    def _initialize(self):

        self.data_available = False
        self.batch = None
        self.thresholds = None
        self.peaks = None
        self.data_lines = None
        self.threshold_lines = None
        self.peak_points = None
        return

    @property
    def nb_channels(self):
        return self.inputs['data'].shape[1]

    @property
    def nb_samples(self):
        return self.inputs['data'].shape[0]

    def _guess_output_endpoints(self):
        pass

    def _process(self):

        self.batch      = self.inputs['data'].receive()
        self.thresholds = self.inputs['mads'].receive(blocking=False)
        peaks           = self.inputs['peaks'].receive(blocking=False)
        if peaks is not None:

            if not self.is_active:
                self._set_active_mode()

            while peaks.pop('offset')/self.nb_samples < self.counter:
                peaks = self.inputs['peaks'].receive()

            self.peaks = peaks

        self.data_available = True

    def _plot(self):
        # Called from the main thread
        pylab.ion()

        if not getattr(self, 'data_available', False):
            return

        self.data_available = False

        if self.data_lines is None:
            self.data_lines = []
            for i in xrange(self.nb_channels):
                offset = self.spacing*i
                self.data_lines.append(pylab.plot(offset + self.batch[:, i], '0.5')[0])
        else:
            for i, line in enumerate(self.data_lines):
                offset = self.spacing*i
                line.set_ydata(offset + self.batch[:, i])

        if self.thresholds is not None:
            if self.threshold_lines is None:
                self.threshold_lines = []
                for i in xrange(self.nb_channels):
                    offset = self.spacing * i
                    self.threshold_lines.append((pylab.plot([0, self.nb_samples], [offset - self.thresholds[i],
                                                                                  offset - self.thresholds[i]],
                                                           'k--')[0],
                                                pylab.plot([0, self.nb_samples], [offset + self.thresholds[i],
                                                                                  offset + self.thresholds[i]],
                                                           'k--')[0]))
            else:
                for i, (lower_line, upper_line) in enumerate(self.threshold_lines):
                    offset = self.spacing * i
                    lower_line.set_ydata([offset - self.thresholds[i], offset - self.thresholds[i]])
                    upper_line.set_ydata([offset + self.thresholds[i], offset + self.thresholds[i]])

        if self.peaks is not None:

            data, channel = zip(*[(self.peaks[key][channel], channel) for key in self.peaks for channel in self.peaks[key]])
            lengths = [len(d) for d in data]
            channel = numpy.repeat(numpy.int_(channel), lengths)
            data = numpy.hstack(data)
            if self.peak_points is None:
                self.peak_points, = pylab.plot(data, self.spacing*channel, 'r.')
            else:
                self.peak_points.set_data(data, self.spacing*channel)


        if self.data_lines is None:
            pylab.xlim(0, self.nb_samples)
            pylab.xlabel('Time [steps]')
            
        pylab.gca().set_title('Buffer %d' %self.counter)
        pylab.draw()
        return