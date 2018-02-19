from circusort.block.block import Block


class Demultiplexer(Block):
    """Demultiplexer"""
    # TODO complete docstring.

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
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        # The following lines are useful to avoid some PyCharms's warnings.
        self.input_specs = self.input_specs
        self.degree = self.degree
        self.overlap = self.overlap
        self.nb_samples = self.nb_samples

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

    def _guess_output_endpoints(self):

        for input_name, structure in zip(self._input_names, self._input_structures):
            if structure == 'array':
                input = self.get_input(input_name)
                for k in range(0, self.degree):
                    output_name = self._get_output_name(input_name, k)
                    output = self.get_output(output_name)
                    output.configure(dtype=input.dtype, shape=input.shape)

        return

    def _process(self):

        # Get tokens (i.e. which outputs should we use?).
        tokens = [
            (self.counter + k) % self.degree
            for k in range(0, self.overlap + 1)
        ]

        for name, policy in zip(self._input_names, self._input_policies):

            if policy == 'hard_blocking':

                data = self.inputs[name].receive(blocking=True)
                for token in tokens:
                    name_ = self._get_output_name(name, token)
                    self.outputs[name_].send(data)

            elif policy == 'soft_blocking':

                if self._is_synced(name):
                    data = self.inputs[name].receive(blocking=True)
                    for token in tokens:
                        name_ = self._get_output_name(name, token)
                        self.outputs[name_].send(data)
                else:
                    data = self.inputs[name].receive(blocking=False)
                    if data is not None:
                        while self._sync_buffer(data, self.nb_samples):
                            data = self.inputs[name].receive(blocking=True)
                        self._set_synced(name)
                        for token in tokens:
                            name_ = self._get_output_name(name, token)
                            self.outputs[name_].send(data)

            elif policy == 'non_blocking':

                data = self.inputs[name].receive(blocking=False)
                if data is not None:
                    for token in tokens:
                        name_ = self._get_output_name(name, token)
                        self.outputs[name_].send(data)

            else:

                raise NotImplementedError()  # TODO complete.

        return

    def _introspect(self):

        pass

        return
