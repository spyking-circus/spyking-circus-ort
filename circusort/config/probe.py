import os
import sys
import logging
import numpy

logger = logging.getLogger(__name__)


class Probe(object):

    def __init__(self, filename, radius=None):
        probe       = {}
        self.path   = os.path.abspath(os.path.expanduser(filename))
        
        if not os.path.exists(self.path):
            logger.error("The probe file %s does not exist" %self.path)
            sys.exit(1)

        try:
            with open(self.path, 'r') as f:
                probetext = f.read()
                exec(probetext, probe)
        except Exception as ex:
            logger.error("Something wrong with the syntax of the probe file:\n" + str(ex))


        assert probe.has_key('channel_groups') == True, logger.error("Something wrong with the syntax of the probe file")

        self.channel_groups = probe['channel_groups']

        key_flags = ['total_nb_channels', 'radius']
        for key in key_flags:
            if not probe.has_key(key):
                logger.error("%s is missing in the probe file" %key)
            setattr(self, key, probe[key])

        if radius is not None:
            self.radius = radius

    def _get_edges(self, i, channel_groups):
        edges = []
        pos_x, pos_y = channel_groups['geometry'][i]
        for c2 in channel_groups['channels']:
            pos_x2, pos_y2 = channel_groups['geometry'][c2]
            if (((pos_x - pos_x2)**2 + (pos_y - pos_y2)**2) <= self.radius**2):
                edges += [c2]
        return edges

    def get_nodes_and_edges(self):
        """
        Retrieve the topology of the probe.
        
        Other parameters
        ----------------
        radius : integer
        
        Returns
        -------
        nodes : ndarray of integers
            Array of channel ids retrieved from the description of the probe.
        edges : dictionary
            Dictionary which link each channel id to the ids of the channels whose
            distance is less or equal than radius.
        
        """
        
        edges  = {}
        nodes  = []

        for key in self.channel_groups.keys():
            for i in self.channel_groups[key]['channels']:
                edges[i] = self._get_edges(i, self.channel_groups[key])
                nodes   += [i]

        return numpy.sort(numpy.array(nodes, dtype=numpy.int32)), edges


    def get_averaged_n_edges(self):
        nodes, edges = self.get_nodes_and_edges(self)
        n = 0
        for key, value in edges.items():
            n += len(value)
        return n/float(len(edges.values()))