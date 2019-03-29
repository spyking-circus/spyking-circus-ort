class Network(object):
    """Network base class.

    Attributes:
        name: string
        params: dictionary
    """

    name = "Network"
    params = {}
    _inputs = {}
    _outputs = {}
    _blocks = {}

    def __init__(self, manager, name=None, log_address=None, log_level=None, **kwargs):

        self.manager = manager
        self.name = self.name if name is None else name
        self.log_address = log_address
        self.log_level = log_level

        self.params.update(kwargs)
        self._configure(**self.params)

        self._create_blocks()

    def _configure(self, **kwargs):

        for key, value in kwargs.items():
            self.params[key] = kwargs[key]
            self.__setattr__(key, value)

        return

    def _create_block(self, *args, **kwargs):

        block = self.manager.create_block(*args, **kwargs)

        return block

    def _create_blocks(self):

        raise NotImplementedError()

    def _add_input(self, name, endpoint):

        self._inputs[name] = endpoint

        return

    def get_input(self, name):

        input_ = self._inputs[name]

        return input_

    def _add_output(self, name, endpoint):

        self._outputs[name] = endpoint

        return

    def get_output(self, name):

        output = self._outputs[name]

        return output

    def _add_block(self, name, block):

        self._blocks[name] = block

        return

    def get_block(self, name):

        block = self._blocks[name]

        return block

    def _connect(self):

        raise NotImplementedError()

    def connect(self):

        self._connect()

        return
