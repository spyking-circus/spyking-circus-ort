from circusort.net.network import Network


__classname__ = "Cluster"


class Cluster(Network):
    """Cluster network.

    Attributes:
        degree: integer
            The number of clustering blocks to use in parallel.
    See also:
        circusort.net.network.Network
    """

    name = "Filter network"

    params = {
        'degree': 2,
        'nb_channels' : 10
    }

    def __init__(self, *args, **kwargs):
        """Initialize filter network.

        Arguments:
            degree: integer (optional)
                The number of filter blocks to use in parallel.
                The default value is 2.
        """

        Network.__init__(self, *args, **kwargs)

        # The following lines are useful to avoid some PyCharm's warnings.
        self.degree = self.degree
        self.nb_channels = self.nb_channels

    def _create_blocks(self):
        """Create the blocks of the network."""

        # # Keyword arguments of filter blocks.
        for k in range(0, self.degree):
            cluster_kwargs[k].update({
                key: value
                for key, value in self.params.items()
                if key in ['introspection_path']
            })

        clusters = {
            k: self._create_block('density_clustering', **cluster_kwargs[k])
            for k in range(0, self.degree)
        }

        # Register network inputs, outputs and blocks.
        for k in range(self.degree):
            self._add_input('data', clusters[k].get_input('data'))
    
        self._add_output('data', grouper.get_output('data'))
        self._add_block('filters', clusters)

        return

    @staticmethod
    def _get_name(name, k):

        name = "{}_{}".format(name, k)

        return name

    def _connect(self):

        clusters = self.get_block('clusters')

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
