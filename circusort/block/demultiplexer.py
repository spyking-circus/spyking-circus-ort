import numpy as np
import sys

from circusort.block.block import Block

if sys.version_info.major == 3:
    unicode = str  # Python 3 compatibility.


__classname__ = 'Demultiplexer'


class Demultiplexer(Block):
    """Demultiplexer.

    Attributes:
        input_specs
        degree: integer
        overlap: integer
        nb_samples: integer
        sampling_rate: float
    """

    name = "Demultiplexer"

    params = {
        'input_specs': [
            {
                'name': "data",
                'structure': 'array',
                'policy': 'hard_blocking',
            }
        ],
        # or
        # 'input_specs': ["data"],
        'degree': 2,
        'overlap': 1,
        'nb_samples': 1024,
        'sampling_rate': 20e+3,  # Hz
    }

    def __init__(self, **kwargs):
        """Initialize block."""

        Block.__init__(self, **kwargs)

        # The following lines are useful to avoid some PyCharms's warnings.
        self.input_specs = self.input_specs
        self.degree = self.degree
        self.overlap = self.overlap
        self.nb_samples = self.nb_samples
        self.sampling_rate = self.sampling_rate

        assert self.degree >= self.overlap  # TODO add docstring.

        # Declare input names and structures.
        self._input_names = []
        self._input_structures = []
        self._input_policies = []
        self._are_synced_inputs = {}
        for input_spec in self.input_specs:
            if isinstance(input_spec, (str, unicode)):
                name = input_spec
                structure = 'array'
                policy = 'hard_blocking'
            elif isinstance(input_spec, dict):
                name = input_spec['name']
                if 'structure' in input_spec:
                    structure = input_spec['structure']
                else:
                    structure = 'array'
                if 'policy' in input_spec:
                    policy = input_spec['policy']
                else:
                    policy = 'hard_blocking'
            else:
                raise NotImplementedError()  # TODO complete.
            self._input_names.append(name)
            self._input_structures.append(structure)
            self._input_policies.append(policy)
            self._are_synced_inputs[name] = False

        # Declare the inputs.
        for name, structure in zip(self._input_names, self._input_structures):
            self.add_input(name, structure)
        # Declare the outputs.
        for name, structure in zip(self._input_names, self._input_structures):
            for k in range(0, self.degree):
                output_name = self._get_output_name(name, k)
                self.add_output(output_name, structure)

        self._nb_channels = None
        self._nb_samples = None

    @staticmethod
    def _get_output_name(name, k):

        name = "{}_{}".format(name, k)

        return name

    def _is_synced(self, name):

        is_synced_input = self._are_synced_inputs[name]

        return is_synced_input

    def _set_synced(self, name):

        self._are_synced_inputs[name] = True

        return

    def _initialize(self):

        pass

        return

    def _configure_input_parameters(self, nb_channels=None, nb_samples=None, **kwargs):

        if nb_channels is not None:
            self._nb_channels = nb_channels
        if nb_samples is not None:
            self._nb_samples = nb_samples

    def _get_output_parameters(self):

        params = {
            'nb_channels': self._nb_channels,
            'nb_samples': self._nb_samples,
        }

        return params

    def _guess_output_endpoints(self):

        for input_name, structure in zip(self._input_names, self._input_structures):
            if structure == 'array':
                input_ = self.get_input(input_name)
                for k in range(0, self.degree):
                    output_name = self._get_output_name(input_name, k)
                    output = self.get_output(output_name)
                    output.configure(dtype=input_.dtype, shape=input_.shape)

        return

    def _process(self):

        self._measure_time('start', frequency=100)

        # Get tokens (i.e. which outputs should we use?).
        tokens = [
            (self.counter + k) % self.degree
            for k in range(0, self.overlap + 1)
        ]

        for name, policy in zip(self._input_names, self._input_policies):

            if policy == 'hard_blocking':

                packet = self.inputs[name].receive(blocking=True)
                for token in tokens:
                    name_ = self._get_output_name(name, token)
                    self.outputs[name_].send(packet)

            elif policy == 'soft_blocking':

                if self._is_synced(name):
                    packet = self.inputs[name].receive(blocking=True)
                    for token in tokens:
                        name_ = self._get_output_name(name, token)
                        self.outputs[name_].send(packet)
                else:
                    packet = self.inputs[name].receive(blocking=False)
                    if packet is not None:
                        data = packet['payload']
                        while not self._sync_buffer(data, self.nb_samples):
                            packet = self.inputs[name].receive(blocking=True)
                            data = packet['payload']
                        self._set_synced(name)
                        for token in tokens:
                            name_ = self._get_output_name(name, token)
                            self.outputs[name_].send(packet)

            elif policy == 'non_blocking':

                packet = self.inputs[name].receive(blocking=False)
                if packet is not None:
                    for token in tokens:
                        name_ = self._get_output_name(name, token)
                        self.outputs[name_].send(packet)

            elif policy == 'non_blocking_broadcast':

                packet = self.get_input(name).receive(blocking=False)
                if packet is not None:
                    for token in range(0, self.degree):
                        output_name = self._get_output_name(name, token)
                        self.get_output(output_name).send(packet)

            else:

                raise NotImplementedError()  # TODO complete.

        self._measure_time('end', frequency=100)

        return

    def _introspect(self):
        """Introspection of the demultiplexer."""

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
