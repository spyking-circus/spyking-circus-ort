class Network(object):
    """Network base class.

    Attributes:
        name: string
        params: dictionary
    """
    # TODO complete docstring.

    name = "Network"
    params = {}
    _inputs = {}
    _outputs = {}
    _blocks = {}

    def __init__(self, manager, name=None, log_address=None, log_level=None, **kwargs):
        # TODO add docstring.

        self.manager = manager
        self.name = self.name if name is None else name
        self.log_address = log_address
        self.log_level = log_level

        self.params.update(kwargs)
        self._configure(**self.params)

        self._create_blocks()

    def _configure(self, **kwargs):
        # TODO add docstring.

        for key, value in kwargs.items():
            self.params[key] = kwargs[key]
            self.__setattr__(key, value)

        return

    def _create_block(self, *args, **kwargs):
        # TODO add docstring.

        block = self.manager.create_block(*args, **kwargs)

        return block

    def _create_blocks(self):
        # TODO add docstring.

        raise NotImplementedError()

    def _add_input(self, name, endpoint):
        # TODO add docstring.

        self._inputs[name] = endpoint

        return

    def get_input(self, name):

        input_ = self._inputs[name]

        return input_

    def _add_output(self, name, endpoint):
        # TODO add docstring.

        self._outputs[name] = endpoint

        return

    def get_output(self, name):
        # TODO add docstring.

        output = self._outputs[name]

        return output

    def _add_block(self, name, block):
        # TODO add docstring.

        self._blocks[name] = block

        return

    def get_block(self, name):
        # TODO add docstring.

        block = self._blocks[name]

        return block

    def _connect(self):
        # TODO add docstring.

        raise NotImplementedError()

    def connect(self):
        # TODO add docstring.

        self._connect()

        return
