# -*- coding: utf-8 -*-

import matplotlib.patches as ptc
import matplotlib.pyplot as plt
# import matplotlib.textpath as ttp
import numpy as np
import os

from matplotlib.collections import PatchCollection


class Probe(object):
    """Open probe file.

    Attributes:
        channel_groups: dictionary
        total_nb_channels: integer
        radius: float
    """

    def __init__(self, channel_groups, total_nb_channels, radius, electrode_diameter=8.0):
        """Initialization.

        Parameters:
            channel_groups: dictionary
            total_nb_channels: integer
            radius: float
            electrode_diameter: float
                The diameter of the electrodes.
                The default value is 8.0.
        """

        self.channel_groups = channel_groups
        self.total_nb_channels = total_nb_channels
        self.radius = radius

        self._electrode_diameter = electrode_diameter  # µm

        self._edges = None
        self._nodes = None
        self._mode = None
        self._path = None
        self._nb_channels = None

    def __len__(self):

        return self._nb_channels

    def _get_edges(self, i, channel_groups):

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
        for channel_group in self.channel_groups.values():
            for channel in channel_group['channels']:
                x_, _ = channel_group['geometry'][channel]
                x.append(x_)
        x = np.array(x)

        return x

    @property
    def y(self):

        y = []
        for channel_group in self.channel_groups.values():
            for channel in channel_group['channels']:
                _, y_ = channel_group['geometry'][channel]
                y.append(y_)
        y = np.array(y)

        return y

    @property
    def channel_nbs(self):

        assert len(self.channel_groups) == 1
        channel_nbs = np.array([
            channel_nb
            for group in self.channel_groups.values()
            for channel_nb in group['channels']
        ])

        return channel_nbs

    @property
    def labels(self):

        if len(self.channel_groups) == 1:
            labels = [
                str(channel)
                for group in self.channel_groups.values()
                for channel in group['channels']
            ]
        elif len(self.channel_groups) > 1:
            labels = [
                str(group) + "/" + str(channel)
                for group in self.channel_groups.values()
                for channel in group['channels']
            ]
        else:
            raise NotImplementedError()

        return labels

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
            positions = np.hstack((positions, np.array(list(self.channel_groups[key]['geometry'].values())).T))
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

    @property
    def x_limits(self, pad=None):
        """Get the x limits of the probe.

        This method is useful to easily set the limits of any matplotlib's figure involving this probe.

        Parameter:
            pad: none | float (optional)
                The size of the pad [µm]. The default value is None.
        Returns:
            x_min: float
            x_max: float
        """

        if pad is None:
            pad = self.minimum_interelectrode_distance
        x_min = self.field_of_view['x_min'] - pad
        x_max = self.field_of_view['x_max'] + pad

        return x_min, x_max
    
    @property
    def y_limits(self, pad=None):
        """Get the y limits of the probe.

        This method is useful to easily set the limits of any matplotlib's figure involving this probe.

        Parameter:
            pad: none | float (optional)
                The size of the pad [µm]. The default value is None.
        Returns:
            y_min: float
            y_max: float
        """

        if pad is None:
            pad = self.minimum_interelectrode_distance
        y_min = self.field_of_view['y_min'] - pad
        y_max = self.field_of_view['y_max'] + pad

        return y_min, y_max

    @property
    def minimum_interelectrode_distance(self):

        d = None
        x = self.x
        y = self.y
        assert x.size == y.size
        n = x.size
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                _dx = x[j] - x[i]
                _dy = y[j] - y[i]
                _d = np.sqrt(np.square(_dx) + np.square(_dy))
                if d is None or _d < d:
                    d = _d
        if d is None:
            d = 0.0

        return d

    def sample_visible_position(self):

        fov = self.field_of_view
        x_min = fov['x_min']
        x_max = fov['x_max']
        y_min = fov['y_min']
        y_max = fov['y_max']
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)

        return x, y

    def get_averaged_n_edges(self):
        n = 0
        for key, value in self.edges.items():
            n += len(value)
        return n/float(len(self.edges.values()))

    def get_channel_position(self, channel, group=None):
        """Get the position (i.e. coordinate) of a channel.

        Parameters:
            channel: integer
            group: integer
        Returns:
            x: float
                The x-coordinate of the channel [µm].
            y: float
                The y-coordinate of the channel [µm].
        """

        _channel = channel

        if group is None:
            _groups = self.channel_groups.keys()
            _group = next(iter(_groups))
        else:
            _group = str(group)
        _group = self.channel_groups[_group]

        _channels = _group['channels']
        _string = "channel {} not found among channels {}"
        _message = _string.format(type(_channel), type(_channels[0]))
        assert _channel in _channels, _message

        _geometry = _group['geometry']
        _position = np.array(_geometry[_channel])
        x = _position[0]
        y = _position[1]

        return x, y

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
        for channel_group_id, channel_group in self.channel_groups.items():
            line = " {}: {{\n".format(channel_group_id)
            lines.append(line)
            line = "  'channels': [{}],\n".format(", ".join([str(k) for k in channel_group['channels']]))
            lines.append(line)
            line = "  'graph': {},\n".format(channel_group['graph'])
            lines.append(line)
            line = "  'geometry': {\n"
            lines.append(line)
            for key, value in channel_group['geometry'].items():
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

    def plot(self, path=None, fig=None, ax=None, annotation_size=None, colors=None, **kwargs):
        """Plot probe.

        Arguments:
            path: none | string (optional)
            fig: none | matplotlib.figure.Figure (optional)
            ax: none | matplotlib.axes.Axes (optional)
            annotation_size: none | float | string (optional)
                Font size of the annotations. Maybe either None (default font size), a size string (relative to the
                default font size), or an absolute font size in points.
                The default value is None.
            colors: none | iterable (optional)
                The colors of the tips of the electrodes.
                The default value is None.
            kwargs: dictionary (optional)
        """

        _ = kwargs  # Discard additional keyword arguments.

        x = self.x
        y = self.y
        r = self._electrode_diameter / 2.0  # µm
        s = self.labels

        if path is not None and ax is None:
            plt.ioff()

        if ax is None:
            fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(*self.x_limits)
        ax.set_ylim(*self.y_limits)
        # Draw the tips of the electrodes.
        if colors is None:
            circles = [
                ptc.Circle((_x, _y), radius=r, color='C0')
                for _x, _y in zip(x, y)
            ]
        else:
            circles = [
                ptc.Circle((x[k], y[k]), radius=r, color=colors[k])
                for k in range(0, self.nb_channels)
            ]
        collection = PatchCollection(circles, match_original=True)
        ax.add_collection(collection)
        # Draw the labels of the electrodes.
        for _x, _y, _s in zip(x, y, s):
            ax.annotate(_s, xy=(_x, _y), xytext=(_x, _y), size=annotation_size,
                        horizontalalignment='center', verticalalignment='center')
        # Draw the default spatial extent of the templates.
        circle = ptc.Circle((np.mean(self.x_limits), np.mean(self.y_limits)), radius=self.radius,
                            fill=False, color='grey', linestyle='--')
        collection = PatchCollection([circle], match_original=True, zorder=-1)
        ax.add_collection(collection)
        # Add labels and title.
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_title("Spatial layout of the electrodes")

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

    def copy(self):
        """Copy probe.

        Return:
            copied_probe: circusort.obj.Probe
                The copied probe.
        """

        copied_probe = Probe(self.channel_groups, self.total_nb_channels, self.radius)

        return copied_probe

    def restrict(self, selection):
        """Restrict to the specified channels.

        Argument:
            selection: iterable [ dictionary
                A data structure which describe the channels to restrict the probe to.
        """

        # Adapt input argument (if necessary).
        if not isinstance(selection, dict):
            nb_channel_group = len(self.channel_groups)
            assert nb_channel_group == 1, "nb_channel_groups: {}".format(nb_channel_group)
            selection = {
                group_key: selection
                for group_key in self.channel_groups
            }

        selected_channel_groups = {}

        for group_key in selection:
            assert group_key in self.channel_groups, "group_key: {}".format(group_key)
            selected_channels = selection[group_key]
            kept_channels = []
            graph = self.channel_groups[group_key]['graph']
            geometry = self.channel_groups[group_key]['geometry']
            assert graph == [], "graph: {}".format(graph)
            for k, channel_key in enumerate(selected_channels):
                assert channel_key in geometry, "channel_key: {}".format(channel_key)
                kept_channels.append(channel_key)
            selected_channel_groups[group_key] = {
                'channels': kept_channels,
                'graph': graph,
                'geometry': geometry,
            }

        self.channel_groups = selected_channel_groups

        return

    def keep(self, selection):
        """Keep the specified channels only.

        Argument:
            selection: none | iterable | dictionary
                A data structure which describe the channels to keep.
        """

        if selection is None:

            pass

        else:

            # Adapt input argument (if necessary).
            if not isinstance(selection, dict):
                nb_channel_groups = len(self.channel_groups)
                assert nb_channel_groups == 1, "nb_channel_groups: {}".format(nb_channel_groups)
                selection = {
                    group_key: selection
                    for group_key in self.channel_groups
                }

            selected_channel_groups = {}
            total_nb_selected_channels = 0

            for group_key in selection:
                assert group_key in self.channel_groups, "group_key: {}".format(group_key)
                selected_channels = selection[group_key]
                kept_channels = []
                kept_graph = []
                kept_geometry = {}
                channels = self.channel_groups[group_key]['channels']
                graph = self.channel_groups[group_key]['graph']
                geometry = self.channel_groups[group_key]['geometry']
                assert graph == [], "graph: {}".format(graph)
                for k, channel_key in enumerate(selected_channels):
                    assert channel_key in channels, "channel_key: {}".format(channel_key)
                    total_nb_selected_channels += 1
                    kept_channels.append(k)
                    kept_geometry[k] = geometry[channel_key]
                selected_channel_groups[group_key] = {
                    'channels': kept_channels,
                    'graph': kept_graph,
                    'geometry': kept_geometry,
                }

            self.channel_groups = selected_channel_groups
            self.total_nb_channels = total_nb_selected_channels

        return

    def get_channel_colors(self, selection=None):

        if selection is None:
            selection = range(0, self.nb_channels)

        channel_colors = self.nb_channels * ['grey']
        for channel in selection:
            channel_colors[channel] = 'C{}'.format(channel % 10)

        return channel_colors
