import os

from ..obj import Peaks


def load_peaks(path):
    # TODO add docstring.

    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        message = "No such peaks file: {}".format(path)
        raise IOError(message)

    peaks = Peaks(path)

    return peaks
