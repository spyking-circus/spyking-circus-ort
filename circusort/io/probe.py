# -*- coding: utf-8 -*-

import os
import sys
import logging

from circusort.obj.probe import Probe


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
    elif len(path) > 0 and path[0] is '/':
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


def generate_probe(nb_electrodes_width=4, nb_electrodes_height=4, interelectrode_distance=30.0, **kwargs):
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

    _ = kwargs  # Discard additional keyword arguments.

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
    # TODO add docstring.

    if isinstance(path, (str, unicode)):
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        if os.path.isdir(path):
            path = os.path.join(path, "probe.prb")
        if os.path.isfile(path):
            # TODO add try ... except ...
            probe = load_probe(path)
        else:
            probe = generate_probe(**kwargs)
    else:
        probe = generate_probe(**kwargs)

    return probe
