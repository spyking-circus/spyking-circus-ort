from circusort.net.network import Network


__classname__ = "Filter"


class Filter(Network):
    """Filter network.

    Attributes:
        degree: integer
            The number of filter blocks to use in parallel.
        cut_off: float
            The cutoff frequency used to define the high-pass filter [Hz].
        order: integer
            The order used to define the high-pass filter.
    See also:
        circusort.net.network.Network
    """

    name = "Filter network"

    params = {
        'degree': 2,
        'cut_off': 500.0,  # Hz
        'order': 1,
    }

    def __init__(self, *args, **kwargs):
        """Initialize filter network.

        Arguments:
            degree: integer (optional)
                The number of filter blocks to use in parallel.
                The default value is 2.
            cut_off: float (optional)
                The cutoff frequency used to define the high-pass filter [Hz].
                The default value is 500.0.
            order: integer (optional)
                The order used to define the high-pass filter.
                The default value is 1.0.
        """

        Network.__init__(self, *args, **kwargs)

        # The following lines are useful to avoid some PyCharm's warnings.
        self.degree = self.degree
        self.cut_off = self.cut_off
        self.order = self.order

    def _create_blocks(self):
        """Create the blocks of the network."""

        # Define keyword arguments.
        # # Keyword arguments of dispatcher block.
        dispatcher_kwargs = {
            'name': 'filter_dispatcher',
            'nb_groups': self.degree,
            'log_level': self.log_level,
        }
        dispatcher_kwargs.update({
            key: value
            for key, value in self.params.items()
            if key in ['introspection_path']
        })
        # # Keyword arguments of filter blocks.
        filter_kwargs = {
            k: {
                'name': '{}_{}'.format(self.name, k),
                'cut_off': self.cut_off,
                'order': self.order,
                'remove_median': False,
                'use_gpu': False,
                'log_level': self.log_level,
            }
            for k in range(0, self.degree)
        }
        for k in range(0, self.degree):
            filter_kwargs[k].update({
                key: value
                for key, value in self.params.items()
                if key in ['introspection_path']
            })
        # # Keyword arguments of grouper block.
        grouper_kwargs = {
            'name': 'filter_grouper',
            'nb_groups': self.degree,
            'log_level': self.log_level,
        }
        grouper_kwargs.update({
            key: value
            for key, value in self.params.items()
            if key in ['introspection_path']
        })

        # Create blocks.
        dispatcher = self._create_block('channel_dispatcher', **dispatcher_kwargs)
        filters = {
            k: self._create_block('filter', **filter_kwargs[k])
            for k in range(0, self.degree)
        }
        grouper = self._create_block('channel_grouper', **grouper_kwargs)

        # Register network inputs, outputs and blocks.
        self._add_input('data', dispatcher.get_input('data'))
        self._add_output('data', grouper.get_output('data'))
        self._add_block('dispatcher', dispatcher)
        self._add_block('filters', filters)
        self._add_block('grouper', grouper)

        return

    @staticmethod
    def _get_name(name, k):

        name = "{}_{}".format(name, k)

        return name

    def _connect(self):

        dispatcher = self.get_block('dispatcher')
        filters = self.get_block('filters')
        grouper = self.get_block('grouper')

        for k in range(0, self.degree):
            # Extract k-th filter.
            filter_ = filters[k]
            # Connect degrouper to k-th filter.
            input_name = 'data'
            output_name = self._get_name(input_name, k)
            self.manager.connect(
                dispatcher.get_output(output_name),
                [filter_.get_input(input_name)]
            )

        for k in range(0, self.degree):
            # Extract k-th filter.
            filter_ = filters[k]
            # Connect k-th filter to grouper.
            output_name = 'data'
            input_name = self._get_name(output_name, k)
            self.manager.connect(
                filter_.get_output(output_name),
                [grouper.get_input(input_name)]
            )

        return
