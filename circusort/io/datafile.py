import numpy as np
import os

from circusort.obj.datafile import DataFile


def load_datafile(path, sampling_rate, nb_channels, dtype, gain=1.0, offset=0, nb_replay=1):
    """Load datafile from path.

    Arguments:
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
        offste: int (optional)
            The offset if the file has a header
    Return:
        train: numpy.array
            Train. An array of spike times.
    """

    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        message = "No such data file: {}".format(path)
        raise IOError(message)

    datafile = DataFile(path, sampling_rate, nb_channels, dtype=dtype, gain=gain, offset=offset, nb_replay=nb_replay)

    return datafile


def create_datafile(path, nb_samples, nb_channels, sampling_rate, dtype, gain=1.0):
    """Create a new data file.

    Arguments:
        path: string
            The path to save the new file.
        nb_samples: integer
            The number of samples.
        nb_channels: integer
            The number of channels.
        sampling_rate: float
            The sampling rate.
        dtype: string
            The data type.
        gain: float (optional)
            The default value is 1.0.
    Return:
        datafile: circusort.obj.Datafile
            The created data file.
    """

    # Create directory (if necessary).
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    mode = 'w+'
    shape = (nb_samples, nb_channels)
    data = np.memmap(path, dtype=dtype, mode=mode, shape=shape)
    del data

    datafile = DataFile(path, sampling_rate, nb_channels, dtype=dtype, gain=gain)

    return datafile
