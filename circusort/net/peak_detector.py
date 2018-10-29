from circusort.net.network import Network


__classname__ = "PeakDetector"


class PeakDetector(Network):
    """Peak detector network.

    Attributes:
        degree: integer
            The number of peak detector blocks to use in parallel.
        threshold_factor: float
            The factor used to define the threshold value based on the standard deviations of the signals.
        sign_peaks: string
            The sign of the peaks to detect (either 'negative', 'positive' or 'both').
        spike_width: float
            The temporal width [ms] of the peaks to be detected.
        sampling_rate: float
            The sampling rate [Hz] of the recorded data.
        safety_time: string
            The minimal time [ms] between two peaks detected consecutively.
    See also:
        circusort.net.network.Network
    """

    name = "Peak detector network"

    params = {
        'degree': 2,
        'threshold_factor': 5.0,
        'sign_peaks': 'negative',
        'spike_width': 5.0,  # ms
        'sampling_rate': 20e+3,  # Hz
        'safety_time': 'auto',
    }

    def __init__(self, *args, **kwargs):
        """Initialize peak detector network.

        Arguments:
            degree: integer (optional)
                The number of peak detector blocks to use in parallel.
                The default value is 2.
            threshold_factor: float (optional)
                The factor used to define the threshold value based on the standard deviations of the signals.
                The default value is 5.0.
            sign_peaks: string (optional)
                The sign of the peaks to detect (either 'negative', 'positive' or 'both").
                The default value is 'negative'.
            spike_width: float (optional)
                The temporal width [ms] of the peaks to be detected.
                The default value is 5.0.
            sampling_rate: float (optional)
                The sampling rate [Hz] of the recorded data.
                The default value is 20e+3.
            safety_time: float | string (optional)
                The minimal time [ms] between two peaks detected consecutively.
                The default value is 'auto'.
        """

        Network.__init__(self, *args, **kwargs)

        # The following lines are useful to avoid some PyCharm's warnings.
        self.degree = self.degree
        self.threshold_factor = self.threshold_factor
        self.sign_peaks = self.sign_peaks
        self.spike_width = self.spike_width
        self.sampling_rate = self.sampling_rate
        self.safety_time = self.safety_time

    def _create_blocks(self):
        """Create the blocks of the network."""

        # Define keyword arguments.
        data_dispatcher_kwargs = {
            'name': 'data_dispatcher',
            'nb_groups': self.degree,
            'log_level': self.log_level,
        }
        data_dispatcher_kwargs.update({
            key: value
            for key, value in self.params.items()
            if key in ['introspection_path']
        })
        mad_dispatcher_kwargs = {
            'name': 'mad_dispatcher',
            'nb_groups': self.degree,
            'log_level': self.log_level,
        }
        mad_dispatcher_kwargs.update({
            key: value
            for key, value in self.params.items()
            if key in ['introspection_path']
        })
        peak_detector_kwargs = {
            k: {
                'name': '{}_{}'.format(self.name, k),
                'threshold_factor': self.threshold_factor,
                'sign_peaks': self.sign_peaks,
                'spike_width': self.spike_width,
                'sampling_rate': self.sampling_rate,
                'safety_time': self.safety_time,
            }
            for k in range(0, self.degree)
        }
        for k in range(0, self.degree):
            peak_detector_kwargs[k].update({
                key: value
                for key, value in self.params.items()
                if key in ['introspection_path']
            })
        peak_grouper_kwargs = {
            'name': 'peak_grouper',
            'nb_groups': self.degree,
            'log_level': self.log_level,
        }
        peak_grouper_kwargs.update({
            key: value
            for key, value in self.params.items()
            if key in ['introspection_path']
        })

        # Create blocks.
        data_dispatcher = self._create_block('channel_dispatcher', **data_dispatcher_kwargs)
        mad_dispatcher = self._create_block('channel_dispatcher', **mad_dispatcher_kwargs)
        detectors = {
            k: self._create_block('peak_detector', **peak_detector_kwargs[k])
            for k in range(0, self.degree)
        }
        peak_grouper = self._create_block('peak_grouper', **peak_grouper_kwargs)

        # Register network inputs, outputs and blocks.
        self._add_input('data', data_dispatcher.get_input('data'))
        self._add_input('mads', mad_dispatcher.get_input('data'))
        self._add_output('peaks', peak_grouper.get_output('peaks'))
        self._add_block('data_dispatcher', data_dispatcher)
        self._add_block('mad_dispatcher', mad_dispatcher)
        self._add_block('detectors', detectors)
        self._add_block('peak_grouper', peak_grouper)

        return

    def _connect(self):

        data_dispatcher = self.get_block('data_dispatcher')
        mad_dispatcher = self.get_block('mad_dispatcher')
        detectors = self.get_block('detectors')
        peak_grouper = self.get_block('peak_grouper')

        for k in range(0, self.degree):
            self.manager.connect(data_dispatcher.get_output('data_{}'.format(k)), [
                detectors[k].get_input('data')
            ])
            self.manager.connect(mad_dispatcher.get_output('data_{}'.format(k)), [
                detectors[k].get_input('mads')
            ])
        for k in range(0, self.degree):
            self.manager.connect(detectors[k].get_output('peaks'), [
                peak_grouper.get_input('peaks_{}'.format(k))
            ])

        return
