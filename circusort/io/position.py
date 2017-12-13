import h5py
import numpy as np
import os

from ..obj import Position


def generate_position(**kwargs):
    """Generate position."""

    # TODO improve this generation.

    _ = kwargs
    x = np.array([0.0])
    y = np.array([0.0])
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

    if path is None:
        position = generate_position(**kwargs)
    elif not os.path.isfile(path):
        position = generate_position(**kwargs)
    else:
        try:
            position = load_position(path)
        except OSError:
            position = generate_position(**kwargs)

    return position
