import numpy as np

from circusort.block.block import Block


class Multiplexer(Block):
    """Multiplexer"""
    # TODO complete docstring.

    name = "Multiplexer"

    params = {
        'output_specs': [
            {
                'name': "data",
                'structure': 'array',
            }
        ],
        # or
        # 'output_specs': ["data"],
        'degree': 2,
        'nb_samples': 1024,
        'sampling_rate': 20e+3,  # Hz
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        # The following lines are useful to avoid some PyCharms's warnings.
        self.output_specs = self.output_specs
        self.degree = self.degree
        self.nb_samples = self.nb_samples
        self.sampling_rate = self.sampling_rate

        # Declare output names and structures.
        self._output_names = []
        self._output_structures = []
        for output_spec in self.output_specs:
            if isinstance(output_spec, (str, unicode)):
                name = output_spec
                structure = 'array'
            elif isinstance(output_spec, dict):
                name = output_spec['name']
                if 'structure' in output_spec:
                    structure = output_spec['structure']
                else:
                    structure = 'array'
            else:
                raise NotImplementedError()  # TODO complete.
            self._output_names.append(name)
            self._output_structures.append(structure)

        # Declare the inputs.
        for name, structure in zip(self._output_names, self._output_structures):
            for k in range(0, self.degree):
                input_name = self._get_input_name(name, k)
                self.add_input(input_name, structure)
        # Declare the outputs.
        for name, structure in zip(self._output_names, self._output_structures):
            self.add_output(name, structure)

    @staticmethod
    def _get_input_name(name, k):

        input_name = "{}_{}".format(name, k)

        return input_name

    def _initialize(self):

        pass

        return

    def _guess_output_endpoints(self):

        for output_name, structure in zip(self._output_names, self._output_structures):
            if structure == 'array':
                output = self.get_output(output_name)
                if self.degree > 0:
                    # Find the data type and shape of the 1st input.
                    input_name = self._get_input_name(output_name, 0)
                    input_ = self.get_output(input_name)
                    dtype = input_.dtype
                    shape = input_.shape
                    # Check the data types and shapes of the other inputs.
                    for k in range(1, self.degree):
                        input_name = self._get_input_name(output_name, k)
                        input_ = self.get_input(input_name)
                        if input_.dtype != dtype:
                            string = "Different input dtypes for the multiplexer ({} or {})"
                            message = string.format(input_.dtype, dtype)
                            self.log.error(message)
                        if input_.shape != shape:
                            string = "Different input shapes for the multiplexer ({} or {})"
                            message = string.format(input_.shape, shape)
                            self.log.error(message)
                    # Configure the output (if everything is consistent).
                    output.configure(dtype=dtype, shape=shape)
                else:
                    raise NotImplementedError()

        return

    def _process(self):

        # TODO remove try: ... except: ...
        try:

            # Get token (i.e. which input should we use?).
            token = self.counter % self.degree

            # Get data.
            input_data = {}
            for output_name in self._output_names:
                input_name = self._get_input_name(output_name, token)
                input_ = self.inputs[input_name]
                input_data[output_name] = input_.receive(blocking=True)

            # TODO clean the 2 following lines.
            # self._measure_time('start', frequency=100)
            self._measure_time('start', frequency=1)

            # Send data.
            for output_name, data in input_data.iteritems():

                self.outputs[output_name].send(data)

            # TODO clean the 2 following lines.
            # self._measure_time('end', frequency=100)
            self._measure_time('end', frequency=1)

        except Exception as exception:

            string = "Exception: {}"
            message = string.format(exception)
            self.log.debug(message)
            raise exception

        return

    def _introspect(self):
        """Introspection of the demultiplexing."""

        # TODO remove try: ... except: ...
        try:
            nb_buffers = self.counter - self.start_step
            start_times = np.array(self._measured_times.get('start', []))
            end_times = np.array(self._measured_times.get('end', []))
            durations = end_times - start_times
            data_duration = float(self.nb_samples) / self.sampling_rate
            ratios = data_duration / durations

            min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
            mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
            max_ratio = np.max(ratios) if ratios.size > 0 else np.nan
        except Exception as exception:
            string = "Exception: {}"
            message = string.format(exception)
            self.log.debug(message)
            raise exception

        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return
