from ..obj import DataFile
import os

def load_datafile(path, sampling_rate, nb_channels, dtype, gain=1.):
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

    Return:
        train: numpy.array
            Train. An array of spike times.
    """

    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        message = "No such data file: {}".format(path)
        raise IOError(message)

    datafile = DataFile(path, sampling_rate, nb_channels, dtype, gain)

    return datafile