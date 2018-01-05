# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os


class Probe(object):
    """Open probe file.

    Attributes:
        channel_groups: dictionary
        total_nb_channels: integer
        radius: float
    """
    # TODO complete docstring.

    def __init__(self, channel_groups, total_nb_channels, radius):
        """Initialization.

        Parameters:
            channel_groups: dictionary
            total_nb_channels: integer
            radius: float
        """
        # TODO complete docstring.

        self.channel_groups = channel_groups
        self.total_nb_channels = total_nb_channels
        self.radius = radius

        self._edges = None
        self._nodes = None
        self._mode = None
        self._path = None
        self._nb_channels = None

    def _get_edges(self, i, channel_groups):
        # TODO add docstring.

        edges = []
        pos_x, pos_y = channel_groups['geometry'][i]
        for c2 in channel_groups['channels']:
            pos_x2, pos_y2 = channel_groups['geometry'][c2]
            if ((pos_x - pos_x2)**2 + (pos_y - pos_y2)**2) <= self.radius**2:
                edges += [c2]

        return np.array(edges, dtype=np.int32)

    def get_nodes_and_edges(self):
        """Retrieve the topology of the probe.

        Returns:
            nodes: numpy.ndarray
                Array of channel ids retrieved from the description of the probe.
            edges: dictionary
                Dictionary which link each channel id to the ids of the channels whose distance is less or equal than
                radius.
        """

        edges = {}
        nodes = []

        for key in self.channel_groups.keys():
            for i in self.channel_groups[key]['channels']:
                edges[i] = self._get_edges(i, self.channel_groups[key])
                nodes += [i]

        return np.sort(np.array(nodes, dtype=np.int32)), edges

    @property
    def x(self):

        x = []
        for channel_group in self.channel_groups.itervalues():
            for channel in channel_group['channels']:
                x_, _ = channel_group['geometry'][channel]
                x.append(x_)
        x = np.array(x)

        return x

    @property
    def y(self):

        y = []
        for channel_group in self.channel_groups.itervalues():
            for channel in channel_group['channels']:
                _, y_ = channel_group['geometry'][channel]
                y.append(y_)
        y = np.array(y)

        return y

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
        positions = np.zeros((2, 0), dtype=np.float32)
        for key in self.channel_groups.keys():
            positions = np.hstack((positions, np.array(self.channel_groups[key]['geometry'].values()).T))
        return positions

    @property
    def field_of_view(self):
        """Field_of_view of the probe.

        Return:
            fov: dictionary
                Field of view of the probe.
        """

        # Collect the x-coordinate and y-coordinate of each channel.
        x = []
        y = []
        for key in self.channel_groups.keys():
            group = self.channel_groups[key]
            x.extend([group['geometry'][c][0] for c in group['channels']])
            y.extend([group['geometry'][c][1] for c in group['channels']])
        x = np.array(x)
        y = np.array(y)
        # Compute the distance between channels.
        d = float('Inf')
        for i in range(0, len(x)):
            p_1 = np.array([x[i], y[i]])
            for j in range(i + 1, len(x)):
                p_2 = np.array([x[j], y[j]])
                d = min(d, np.linalg.norm(p_2 - p_1))
        # Compute the field of view of the probe.
        fov = {
            'x_min': np.amin(x),
            'y_min': np.amin(y),
            'x_max': np.amax(x),
            'y_max': np.amax(y),
            'd': d,
            'w': np.amax(x) - np.amin(x),
            'h': np.amax(y) - np.amin(y),
        }

        return fov

    def get_averaged_n_edges(self):
        n = 0
        for key, value in self.edges.items():
            n += len(value)
        return n/float(len(self.edges.values()))

    def get_channels_around(self, x, y, r=None):
        """Get channel identifiers around a given point in space.

        Parameters:
            x: float
                x-coordinate.
            y: float
                y-coordinate
            r: none | float (optional)
                Radius [µm]. The default value is None.

        Returns:
            channels: numpy.ndarray
                The channels in the neighborhood of the given point.
            distances: numpy.ndarray
                The distances between each channel and the given point.
        """

        channels = []
        distances = []

        pos = np.array([x, y])
        for key in self.channel_groups.keys():
            channel_group = self.channel_groups[key]
            for channel in channel_group['channels']:
                pos_c = np.array(channel_group['geometry'][channel])
                d = np.linalg.norm(pos_c - pos)
                if r is None or d < r:
                    # Channel position is near given position.
                    channels += [channel]
                    distances += [d]
        channels = np.array(channels, dtype='int')
        distances = np.array(distances, dtype='float')

        return channels, distances

    def save(self, path):
        """Save probe to file.

        Parameter:
            path: string
                Path to which the probe is saved.
        """

        # Normalize path.
        path = os.path.expanduser(path)
        path = os.path.abspath(path)

        # Handle mode.
        if path[-4:] != ".prb":
            path = os.path.join(path, "probe.prb")
        directory = os.path.dirname(path)
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # Prepare lines to be saved.
        lines = []
        # Save total number of channels to probe file.
        line = "total_nb_channels = {}\n".format(self.total_nb_channels)
        lines.append(line)
        # Save radius to probe file.
        line = "radius = {}\n".format(self.radius)
        lines.append(line)
        # Save `channel_groups` to probe file.
        line = "channel_groups = {\n"
        lines.append(line)
        for channel_group_id, channel_group in self.channel_groups.iteritems():
            line = " {}: {{\n".format(channel_group_id)
            lines.append(line)
            line = "  'channels': {},\n".format(channel_group['channels'])
            lines.append(line)
            line = "  'graph': {},\n".format(channel_group['graph'])
            lines.append(line)
            line = "  'geometry': {\n"
            lines.append(line)
            for key, value in channel_group['geometry'].iteritems():
                line = "   {}: {},\n".format(key, value)
                lines.append(line)
            line = "  },\n"
            lines.append(line)
            line = " },\n"
            lines.append(line)
        line = "}\n"
        lines.append(line)
        line = "\n"
        lines.append(line)

        # Open probe file.
        file_ = open(path, mode='w')
        # Write lines to save.
        file_.writelines(lines)
        # Close probe file.
        file_.close()

        # Update private attributes.
        self._mode = 'file'
        self._path = path
        self._nb_channels = self.total_nb_channels

        return

    def plot(self, path=None, fig=None, ax=None, **kwargs):
        # TODO add docstring.

        _ = kwargs  # Discard additional keyword arguments.

        x = self.x
        y = self.y
        x_min = np.amin(x) - 10.0
        x_max = np.amax(x) + 10.0
        y_min = np.amin(y) - 10.0
        y_max = np.amax(y) + 10.0

        if path is not None and ax is None:
            plt.ioff()

        if ax is None:
            fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.scatter(x, y)  # TODO control the radius of the electrodes.
        ax.set_xlabel(u"x (µm)")
        ax.set_ylabel(u"y (µm)")
        ax.set_title(u"Spatial layout of the electrodes")

        if fig is not None and path is None:
            plt.tight_layout()
            plt.show()
        elif fig is not None and path is not None:
            # Normalize path.
            path = os.path.expanduser(path)
            path = os.path.abspath(path)

            # Handle mode.
            if path[-4:] != ".pdf":
                path = os.path.join(path, "probe.pdf")
            directory = os.path.dirname(path)
            if not os.path.isdir(directory):
                os.makedirs(directory)

            plt.tight_layout()
            plt.savefig(path)

        return

    def get_nearest_electrode_distance(self, point):
        """Get distance to nearest electrode.

        Parameter:
            point: tuple
                Point coordinate (e.g. `(0.0, 1.0)`).

        Return:
            distance: float
                Distance to nearest electrode.
        """

        x, y = point
        _, distances = self.get_channels_around(x, y)
        distance = np.min(distances)

        return distance

    def get_electrodes_around(self, point, radius):
        """Get the electrodes in the neighborhood of a given point.

        Parameters:
            point: tuple
                The given point.
            radius: float
                The radius to use to define the neighborhood disc.

        Return:
            electrodes: numpy.ndarray
                The electrodes in the neighborhood of the given point.
        """
        # TODO complete docstring.

        x, y = point
        electrodes, _ = self.get_channels_around(x, y, r=radius)

        return electrodes

    def get_parameters(self):
        """Get the parameters of the probe.

        Return:
              parameters: dictionary
                A dictionary which contains the parameters of the probe.
        """

        if self._path is None:
            parameters = {}
        else:
            parameters = {
                'mode': self._mode,
                'path': self._path,
                'nb_channels': self._nb_channels,
            }
            # TODO use an ordered dictionary instead.
        # TODO add other parameters (i.e. mode, nb_rows, nb_columns, interelectrode_distance).

        return parameters
