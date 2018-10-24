import numpy as np

from circusort.block.block import Block


__classname__ = "PeakGrouper"


class PeakGrouper(Block):
    """Peak grouper.

    Attribute:
        nb_groups: integer
    """

    name = "Peak grouper"

    param = {
        'nb_groups': 1,
    }

    def __init__(self, **kwargs):
        """Initialize peak grouper.

        Argument:
            nb_groups: integer (optional)
                The number of groups from which data will be gathered.
                The default value is 1.
        """

        Block.__init__(self, **kwargs)

        # The following line is useful to avoid some PyCharm's warning.
        self.nb_groups = self.nb_groups

        for k in range(0, self.nb_groups):
            self.add_input('peaks_{}'.format(k), structure='dict')
        self.add_output('peaks', structure='dict')

        self.nb_samples = None
        self.sampling_rate = None
        # TODO define additional parameters (if necessary).

        # TODO define internal parameters (if necessary).

    def _initialize(self):

        pass

        return

    def _configure_input_parameters(self, nb_samples=None, sampling_rate=None, **kwargs):

        if nb_samples is not None:
            self.nb_samples = nb_samples
        if sampling_rate is not None:
            self.sampling_rate = sampling_rate
        # TODO configure additional parameters (if necessary).

        return

    def _update_initialization(self):

        pass

        return

    def _get_output_parameters(self):

        params = {
            # 'nb_samples': self.nb_samples,  # TODO uncomment?
            # 'sampling_rate': self.sampling_rate,  # TODO uncomment?
        }

        return params

    def _process(self):

        # Receive input packets.
        input_packets = {}
        for k in range(0, self.nb_groups):
            input_name = 'peaks_{}'.format(k)
            input_packets[k] = self.get_input(input_name).receive()

        self._measure_time('start')

        # Unpack input packets.
        number = None
        grouped_peaks = {}
        for k in range(0, self.nb_groups):
            number = input_packets[k]['number']
            peaks = input_packets[k]['payload']
            for key, value in peaks.items():
                if key == 'offset':
                    grouped_peaks.update([(key, value)])
                elif key in ['negative', 'positive']:
                    if key in grouped_peaks:
                        grouped_peaks[key].update(value)
                    else:
                        grouped_peaks[key] = value
                else:
                    pass

        # Send output packet.
        output_packet = {
            'number': number,
            'payload': grouped_peaks,
        }
        self.output.send(output_packet)

        self._measure_time('end')

        return

    def _introspect(self):

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self.nb_samples) / self.sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        # Log info message.
        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return
