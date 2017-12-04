# -*- coding: utf-8 -*-

import os
import sys
import logging
import numpy as np


def resolve_probe_path(path, logger=None):
    """Resolve probe path.

    Parameter:
        path: string
            Path to which the probe will be saved.
    """
    # TODO complete docstring.

    # Define logger.
    if logger is None:
        logger = logging.getLogger(__name__)

    if len(path) > 0 and path[0] is '~':
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            message = "No such probe file: {}".format(path)
            logger.error(message)
            sys.exit(1)
    elif len(path > 0 and path[0] is '/'):
        # TODO make this case compatible with Windows.
        if not os.path.isfile(path):
            message = "No such probe file: {}".format(path)
            logger.error(message)
            sys.exit(1)
    else:
        if os.path.isfile(os.path.abspath(path)):
            path = os.path.abspath(path)
        else:
            path = os.path.join("~", ".spyking-circus-ort", "probes", path)
            path = os.path.expanduser(path)
            if not os.path.isfile(path):
                message = "No such probe file: {}".format(path)
                logger.error(message)
                sys.exit(1)

    return path


def generate_probe(nb_electrodes_width=4, nb_electrodes_height=4, interelectrode_distance=30.0):
    """Generate probe

    Parameters:
        nb_electrodes_width: integer
            Number of columns of electrodes. The default value is 4.
        nb_electrodes_height: integer
            Number of rows of electrodes. The default value is 4.
        interelectrode_distance: float
            Interelectrode distance [µm]. The default value is 30.0.

    Return:
        probe: Probe
            Generated probe.
    """

    nb_electrodes = nb_electrodes_width * nb_electrodes_height

    geometry = {}
    x_offset = - 0.5 * float(nb_electrodes_width - 1) * interelectrode_distance
    y_offset = - 0.5 * float(nb_electrodes_height - 1) * interelectrode_distance
    for k in range(0, nb_electrodes):
        x = float(k % nb_electrodes_width) * interelectrode_distance + x_offset  # µm
        y = float(k / nb_electrodes_width) * interelectrode_distance + y_offset  # µm
        geometry[k] = [x, y]

    channel_group = {
        'channels': list(range(nb_electrodes)),
        'graph': [],
        'geometry': geometry,
    }

    probe_kwargs = {
        'total_nb_channels': nb_electrodes,
        'radius': 250.0,  # µm
        'channel_groups': {1: channel_group},
    }

    probe = Probe(**probe_kwargs)

    return probe


def save_probe(path, probe):
    """Save probe to file.

    Parameters:
        path: string
            Path to which the probe is saved.
        probe: Probe
            Probe object to be saved.
    """

    probe.save(path)

    return


def load_probe(path, radius=None, logger=None):
    """Load probe from file.

    Parameter:
        path: string
            Path to which the probe is saved.
    """
    # TODO complete docstring.

    # Resolve path.
    path = resolve_probe_path(path, logger=logger)

    # Read probe.
    probe_kwargs = {}
    try:
        with open(path, mode='r') as probe_file:
            probe_text = probe_file.read()
            exec(probe_text, probe_kwargs)
    except Exception as exception:
        message = "Something wrong with the syntax of the probe file:\n{}".format(str(exception))
        logger.error(message)

    required_keys = [
        'channel_groups',
        'total_nb_channels',
        'radius',
    ]
    for key in required_keys:
        message = "'{}' key is missing in the probe file {}".format(key, path)
        assert key in probe_kwargs, logger.error(message)

    if radius is not None:
        probe_kwargs['radius'] = radius

    probe = Probe(**probe_kwargs)

    return probe


class Probe(object):
    """Open probe file."""
    # TODO: complete docstring.

    def __init__(self, channel_groups=None, total_nb_channels=None, radius=None):

        self._edges = None
        self._nodes = None

        if channel_groups is None:
            raise ValueError("channel_groups is None")
        else:
            self.channel_groups = channel_groups

        if total_nb_channels is None:
            raise ValueError("total_nb_channels is None")
        else:
            self.total_nb_channels = total_nb_channels

        if radius is None:
            raise ValueError("radius is None")
        else:
            self.radius = radius

    def _get_edges(self, i, channel_groups):
        edges = []
        pos_x, pos_y = channel_groups['geometry'][i]
        for c2 in channel_groups['channels']:
            pos_x2, pos_y2 = channel_groups['geometry'][c2]
            if ((pos_x - pos_x2)**2 + (pos_y - pos_y2)**2) <= self.radius**2:
                edges += [c2]
        return np.array(edges, dtype=np.int32)

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

        edges = {}
        nodes = []

        for key in self.channel_groups.keys():
            for i in self.channel_groups[key]['channels']:
                edges[i] = self._get_edges(i, self.channel_groups[key])
                nodes += [i]

        return np.sort(np.array(nodes, dtype=np.int32)), edges

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

        pos = np.array([x, y])
        for key in self.channel_groups.keys():
            channel_group = self.channel_groups[key]
            for channel in channel_group['channels']:
                pos_c = np.array(channel_group['geometry'][channel])
                d = np.linalg.norm(pos_c - pos)
                if d < r:
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

        # Make directories (if necessary).
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
            line = "  'geometry: {\n"
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
        probe_file = open(path, mode='w')

        # Write lines to save.
        probe_file.writelines(lines)

        # Close probe file.
        probe_file.close()

        return
