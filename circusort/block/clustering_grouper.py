
import numpy as np

from circusort.block.block import Block


__classname__ = "ClusteringGrouper"


class ClusteringGrouper(Block):
    """Channel grouper.

    Attribute:
        nb_groups: integer
    """

    name = "Clustering grouper"

    params = {
        'nb_groups': 1,
    }

    def __init__(self, **kwargs):
        """Initialize channel grouper.

        Argument:
            nb_groups: integer (optional)
                The number of groups from which data will be gathered.
                The default value is 1.
        """

        Block.__init__(self, **kwargs)

        # The following line is useful to avoid some PyCharm's warning.
        self.nb_groups = self.nb_groups

        for k in range(0, self.nb_groups):
            self.add_input('templates_{}'.format(k), structure='dict')
        self.add_output('templates', structure='dict')

        self.dtype = None
        self.nb_samples = None
        self.nb_channels = None
        self.sampling_rate = None

        self._result = None

    def _initialize(self):

        pass

        return

    def _configure_input_parameters(self, dtype=None, nb_samples=None, nb_channels=None, sampling_rate=None, **kwargs):

        if dtype is not None:
            self.dtype = dtype
        if nb_samples is not None:
            self.nb_samples = nb_samples
        if nb_channels is not None:
            self.nb_channels = nb_channels
        if sampling_rate is not None:
            self.sampling_rate = sampling_rate

        return

    def _update_initialization(self):

        return

    def _get_output_parameters(self):

        params = {
        }

        return params

    def _process(self):

        # Receive input packets.
        packets = {}
        for k in range(0, self.nb_groups):
            input_name = 'templates_{}'.format(k)
            packets[k] = self.get_input(input_name).receive(blocking=False)

        self._measure_time('start')

        # Unpack input packets.
        number = None
        all_templates = {'templates_' : {}}
        for k in range(0, self.nb_groups):
            number = input_packets[k]['number']
            peaks = input_packets[k]['payload']['peaks']
            offset = input_packets[k]['payload']['offset']

            for key, value in peaks.items():
                if key in ['negative', 'positive']:
                    # Remap channels correctly.
                    value = {
                        str(int(channel) * self.nb_groups + k): times
                        for channel, times in value.items()
                    }
                    # Accumulate peaks.
                    if key in grouped_peaks:
                        grouped_peaks['peaks'][key].update(value)
                    else:
                        grouped_peaks['peaks'][key] = value
                else:
                    pass

            all_templates['offset'] = offset

        all_templates['thresholds'] = np.hstack([input_packets[k]['payload']['thresholds'] for k in range(self.nb_groups)])

        # Send output packet.
        output_packet = {
            'number': number,
            'payload': all_templates
        }
        self.output.send(output_packet)

        self._measure_time('end')

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
