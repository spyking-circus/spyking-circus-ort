# -*- coding: utf-8 -*-

import os
import sys
import logging

from circusort.obj.probe import Probe

if sys.version_info.major == 3:
    unicode = str


def resolve_probe_path(path, logger=None):
    """Resolve probe path.

    Parameter:
        path: string
            Path to which the probe will be saved.
    """

    # Define logger.
    if logger is None:
        logger = logging.getLogger(__name__)

    if len(path) > 0 and path[0] == '~':
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            message = "No such probe file: {}".format(path)
            logger.error(message)
            sys.exit(1)
    elif len(path) > 0 and path[0] == '/':
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


def generate_mea_probe(nb_columns=4, nb_rows=4, interelectrode_distance=30.0, radius=250.0, **kwargs):
    """Generate a multi-electrode array probe.

    Parameters:
        nb_columns: integer
            Number of columns of electrodes. The default value is 4.
        nb_rows: integer
            Number of rows of electrodes. The default value is 4.
        interelectrode_distance: float
            Interelectrode distance [µm]. The default value is 30.0.
        radius: float
            Template radius [µm]. The default value is 250.0.

    Return:
        probe: Probe
            Generated multi-electrode array probe.
    """

    _ = kwargs  # i.e. discard additional keyword arguments

    nb_electrodes = nb_columns * nb_rows

    geometry = {}
    x_offset = - 0.5 * float(nb_columns - 1) * interelectrode_distance
    y_offset = - 0.5 * float(nb_rows - 1) * interelectrode_distance
    for k in range(0, nb_electrodes):
        x = float(k % nb_columns) * interelectrode_distance + x_offset  # µm
        y = float(k // nb_columns) * interelectrode_distance + y_offset  # µm
        geometry[k] = [x, y]

    channel_group = {
        'channels': list(range(nb_electrodes)),
        'graph': [],
        'geometry': geometry,
    }

    probe_kwargs = {
        'total_nb_channels': nb_electrodes,
        'radius': float(radius),  # µm
        'channel_groups': {1: channel_group},
    }

    probe = Probe(**probe_kwargs)

    return probe


def generate_silicon_probe(**kwargs):
    """Generate a multi-electrode array probe."""

    _ = kwargs  # Discard additional keyword arguments.

    raise NotImplementedError()  # TODO complete.


def generate_probe(mode='default', **kwargs):
    """Generate a probe.

    Parameter:
        mode: string
            The mode to use to generate the probe. Either 'default', 'mea' or 'silicon'. The default value is 'default'.

    Return:
        probe: circusort.obj.Probe
            The generated probe.

    See also:
        circusort.io.probe.generate_mea_probe
        circusort.io.probe.generate_silicon_probe
    """

    if mode in ['default', 'mea']:
        probe = generate_mea_probe(**kwargs)
    elif mode in ['silicon']:
        probe = generate_silicon_probe(**kwargs)
    else:
        message = "Unknown mode value: {}".format(mode)
        raise ValueError(message)

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


def load_probe(path, radius=None, logger=None, **kwargs):
    """Load probe from file.

    Parameter:
        path: string
            The path from which to load the probe.
        radius: none | float
            The radius of the signal horizon to associate to the probe. The default value is None.
        logger: none | logging.Logger
            The logger to use while loading the probe.

    Return:
        probe: circusort.obj.Probe
            The loaded probe.
    """

    _ = kwargs  # Discard additional keyword arguments.

    # Resolve path.
    path = resolve_probe_path(path, logger=logger)

    # Read probe.
    probe_kwargs = {}
    try:
        with open(path, mode='r') as probe_file:
            probe_text = probe_file.read()
            exec(probe_text, probe_kwargs)
            del probe_kwargs['__builtins__']
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


def get_probe(path=None, **kwargs):
    """Get probe from path.

    Parameter:
        path: string
            The path to use to get the probe.

    Return:
        probe: circusort.obj.Probe
            The probe to get.

    See also:
        circusort.io.load_probe
        circusort.io.generate_probe
    """

    if isinstance(path, (str, unicode)):
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        if os.path.isdir(path):
            path = os.path.join(path, "probe.prb")
        if os.path.isfile(path):
            try:
                probe = load_probe(path, **kwargs)
            except IOError:
                probe = generate_probe(**kwargs)
        else:
            probe = generate_probe(**kwargs)
    else:
        probe = generate_probe(**kwargs)

    return probe
