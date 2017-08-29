import os
import sys
import logging
import numpy



def resolve_probe_path(path):
    '''Resolve probe path'''

    path = os.path.expanduser(path)

    if os.path.exists(path):
        path = os.path.abspath(path)
    else:
        path = os.path.join("~", ".spyking-circus-ort", "probes", path)
        path = os.path.expanduser(path)

    return path


class Probe(object):
    '''
    Open probe file.

    TODO: complete.
    '''

    def __init__(self, filename, radius=None, logger=None):

        # Define logger.
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        # Resolve input filename.
        self.path = resolve_probe_path(filename)
        if not os.path.exists(self.path):
            self.logger.error("The probe file %s does not exist" %self.path)
            sys.exit(1)

        self._edges = None
        self._nodes = None

        probe = {}
        try:
            with open(self.path, 'r') as f:
                probetext = f.read()
                exec(probetext, probe)
        except Exception as ex:
            self.logger.error("Something wrong with the syntax of the probe file:\n" + str(ex))


        assert probe.has_key('channel_groups') == True, logger.error("Something wrong with the syntax of the probe file")

        self.channel_groups = probe['channel_groups']

        key_flags = ['total_nb_channels', 'radius']
        for key in key_flags:
            if not probe.has_key(key):
                self.logger.error("%s is missing in the probe file" %key)
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
        return numpy.array(edges, dtype=numpy.int32)

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


    @property
    def edges(self):
        if self._edges is None:
            self._nodes, self._edges = self.get_nodes_and_edges()
        return self._edges

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes, self._edges = self.get_nodes_and_edges()
        return self._nodes

    @property
    def nb_channels(self):
        return len(self.nodes)

    @property
    def positions(self):
        positions = numpy.zeros((2, 0), dtype=numpy.float32)
        for key in self.channel_groups.keys():
            positions = numpy.hstack((positions, numpy.array(self.channel_groups[key]['geometry'].values()).T))
        return positions

    @property
    def field_of_view(self):
        """
        Field_of_view of the probe.

        return
        ------
        fov: dict
            Field of view of the probe.
        """

        # Collect the x-coordinate and y-coordinate of each channel.
        x = []
        y = []
        for key in self.channel_groups.keys():
            group = self.channel_groups[key]
            x.extend([group['geometry'][c][0] for c in group['channels']])
            y.extend([group['geometry'][c][1] for c in group['channels']])
        x = numpy.array(x)
        y = numpy.array(y)
        # Compute the distance between channels.
        d = float('Inf')
        for i in range(0, len(x)):
            p_1 = numpy.array([x[i], y[i]])
            for j in range(i + 1, len(x)):
                p_2 = numpy.array([x[j], y[j]])
                d = min(d, numpy.linalg.norm(p_2 - p_1))
        # Compute the field of view of the probe.
        fov = {
            'x_min': numpy.amin(x),
            'y_min': numpy.amin(y),
            'x_max': numpy.amax(x),
            'y_max': numpy.amax(y),
            'd': d,
            'w': numpy.amax(x) - numpy.amin(x),
            'h': numpy.amax(y) - numpy.amin(y),
        }

        return fov

    def get_averaged_n_edges(self):
        n = 0
        for key, value in self.edges.items():
            n += len(value)
        return n/float(len(self.edges.values()))

    def get_channels_around(self, x, y, r):
        """Get channel identifiers around a given point in space

        Parameters
        ----------
        x: float
            x-coordinate.
        y: float
            y-coordinate
        r: float
            Radius in um.

        """

        channels = []
        distances = []

        pos = numpy.array([x, y])
        for key in self.channel_groups.keys():
            channel_group = self.channel_groups[key]
            for channel in channel_group['channels']:
                pos_c = numpy.array(channel_group['geometry'][channel])
                d = numpy.linalg.norm(pos_c - pos)
                if d < r:
                    # Channel position is near given position.
                    channels += [channel]
                    distances +=[d]
        channels = numpy.array(channels, dtype='int')
        distances = numpy.array(distances, dtype='float')

        return channels, distances
