import h5py
import numpy as np
import os


def generate_position(**kwargs):
    """Generate position."""

    # TODO improve this generation.

    _ = kwargs
    x = np.array([0.0])
    y = np.array([0.0])
    position = (x, y)

    return position


def save_position(path, position):
    """Save position to file.

    Parameters:
        path: string
            The path to the file in which to save the position.
        position: numpy.ndarray
            The position to save.
    """

    x, y = position

    file_ = h5py.File(path, mode='w')
    file_.create_dataset('x', shape=x.shape, dtype=x.dtype, data=x)
    file_.create_dataset('y', shape=y.shape, dtype=y.dtype, data=y)
    file_.close()

    return


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

    position = (x, y)

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
