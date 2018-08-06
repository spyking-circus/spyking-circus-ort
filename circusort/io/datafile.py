import numpy as np
import os

from circusort.obj.datafile import DataFile


def load_datafile(path, sampling_rate, nb_channels, dtype, gain=1.0):
    """Load datafile from path.

    Parameter:
        path: string
            Path from which to load the train.
        sampling_rate: float
            The sampling rate
        nb_channels: int
            The total number of channels in the data
        dtype: float
            The data type
        gain: float (optional)
            The data gain.
            The default value is 1.0.

    Return:
        train: numpy.array
            Train. An array of spike times.
    """

    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        message = "No such data file: {}".format(path)
        raise IOError(message)

    datafile = DataFile(path, sampling_rate, nb_channels, dtype=dtype, gain=gain)

    return datafile


def create_datafile(path, nb_samples, nb_channels, sampling_rate, dtype, gain=1.0):

    mode = 'w+'
    shape = (nb_samples, nb_channels)
    data = np.memmap(path, dtype=dtype, mode=mode, shape=shape)
    del data

    datafile = DataFile(path, sampling_rate, nb_channels, dtype=dtype, gain=gain)

    return datafile
