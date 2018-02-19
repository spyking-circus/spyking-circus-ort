# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os

from circusort.obj.position import Position
from circusort.obj.train import Train


def generate_position(x=0.0, y=0.0, train=None, probe=None, **kwargs):
    """Generate position.

    Parameters:
        x: float
            A pregenerated x-coordinate [µm]. The default value is 0.0.
        y: float
            A pregenerated y_coordinate [µm]. The default value is 0.0.
        train: none | circusort.obj.Train
            The times to use to generate the position. The default value is None.
        probe: none | circusort.obj.Probe
            The probe to use to generate the position. The default value is None.
    Return:
        x: float
            The generated x-coordinate [µm].
        y: float
            The generated y-coordinate [µm].
    """

    if x == 0.0 and y == 0.0:
        if probe is not None:
            x, y = probe.sample_visible_position()

    _ = kwargs
    if isinstance(train, Train):
        nb_times = train.nb_times
    else:
        nb_times = 1
    if isinstance(x, float):
        x = x * np.ones(nb_times)
    elif isinstance(x, (str, unicode)):
        f = eval("lambda t: {}".format(x), kwargs)
        f = np.vectorize(f)
        x = f(train.times)
    else:
        message = "Unknown x type: {}".format(type(x))
        raise TypeError(message)
    if isinstance(y, float):
        y = y * np.ones(nb_times)
    elif isinstance(y, str):
        f = eval("lambda t: {}".format(y), kwargs)
        f = np.vectorize(f)
        y = f(train.times)
    else:
        message = "Unknown y type: {}".format(type(y))
        raise TypeError(message)
    position = Position(x, y)

    return position


def save_position(path, position):
    """Save position to file.

    Parameters:
        path: string
            The path to the file in which to save the position.
        position: numpy.ndarray
            The position to save.
    """

    position.save(path)

    return


def list_positions(directory):
    """List position paths contained in the specified directory.

    Parameter:
        directory: string
            Directory from which to list the positions.

    Return:
        paths: list
            List of position paths found in the specified directory.
    """

    if not os.path.isdir(directory):
        message = "No such position directory: {}".format(directory)
        raise OSError(message)

    filenames = os.listdir(directory)
    filenames.sort()
    paths = [
        os.path.join(directory, filename)
        for filename in filenames
    ]

    return paths


def load_position(path):
    """Load position from file.

    Parameter:
        path: string
            The path to the file from which to load the position.

    Return:
        position: tuple
            The loaded position.
    """

    if not os.path.isfile(path):
        message = "No such position file: {}".format(path)
        raise IOError(message)

    file_ = h5py.File(path, mode='r')
    x = file_.get('x').value
    y = file_.get('y').value
    file_.close()

    position = Position(x, y)

    return position


def get_position(path=None, **kwargs):
    """Get position.

    Parameter:
        path: none | string (optional)
            The path to use to get the position. The default value is None.

    Return:
        position: numpy.ndarray
            The position to get.

    See also:
        circusort.io.generate_position (for additional parameters)
    """

    if isinstance(path, (str, unicode)):
        try:
            position = load_position(path)
        except IOError:
            position = generate_position(**kwargs)
    else:
        position = generate_position(**kwargs)

    return position
