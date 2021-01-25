from circusort.net.network import Network
import numpy as np

__classname__ = "DensityClustering"


class DensityClustering(Network):
    """Cluster network.

    Attributes:
        degree: integer
            The number of clustering blocks to use in parallel.
    See also:
        circusort.net.network.Network
    """

    name = "Density Clustering network"

    params = {
        'degree': 4,
        'nb_channels' : 256
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

        dispatcher_kwargs = {
            'name': 'clustering_dispatcher',
            'nb_groups': self.degree,
            'log_level': self.log_level,
        }

        grouper_kwargs = {
            'name' : 'clutering_grouper',
            'nb_groups' : self.degree,
            'log_level' : self.log_level
        }

        cluster_kwargs = {k : self.params.copy() for k in range(self.degree)}

        for k in range(0, self.degree):
            channels = list(range(k, self.nb_channels, self.degree))
            cluster_kwargs[k].update({
                'channels' : channels
            })

        clusters = {
            k: self._create_block('density_clustering', **cluster_kwargs[k])
            for k in range(0, self.degree)
        }

        clustering_dispatcher = self._create_block('clustering_dispatcher', **dispatcher_kwargs)        
        clustering_grouper = self._create_block('clustering_grouper', **grouper_kwargs)

        self._add_input('data', clustering_dispatcher.get_input('data'))
        self._add_input('peaks', clustering_dispatcher.get_input('peaks'))
        self._add_input('pcs', clustering_dispatcher.get_input('pcs'))

        self._add_output('templates', clustering_grouper.get_output('templates'))

        self._add_block('clustering_dispatcher', clustering_dispatcher)
        self._add_block('clusters', clusters)
        self._add_block('clustering_grouper', clustering_grouper)

        return

    @staticmethod
    def _get_name(name, k):

        name = "{}_{}".format(name, k)

        return name

    def _connect(self):

        clusters = self.get_block('clusters')
        clustering_dispatcher = self.get_block('clustering_dispatcher')
        clustering_grouper = self.get_block('clustering_grouper')

        for name in ['data', 'peaks', 'pcs']:
            self.manager.connect(
                clustering_dispatcher.get_output(name),
                [clusters[k].get_input(name) for k in range(self.degree)]
            )

        for k in range(self.degree):
            self.manager.connect(
                clusters[k].get_output('templates'),
                clustering_grouper.get_input('templates_%d' %k),

            )

        return
