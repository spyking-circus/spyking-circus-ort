import os

from ..obj import MADFile


def load_madfile(path, dtype, nb_channels, nb_samples_per_buffer, sampling_rate):
    # TODO add docstring.

    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        message = "No such MAD file: {}".format(path)
        raise IOError(message)

    madfile = MADFile(path, dtype, nb_channels, nb_samples_per_buffer, sampling_rate)

    return madfile
