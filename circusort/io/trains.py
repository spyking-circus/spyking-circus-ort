import h5py
import numpy as np
import os
import sys

from ..obj import Train

if sys.version_info.major == 3:
    unicode = str


def generate_train(duration=60.0, rate=1.0, **kwargs):
    """Generate train.

    Parameters:
        duration: float (optional)
            Train duration [s]. The default value is 60.0.
        rate: float (optional)
            Spike rate [Hz]. The default value is 1.0.

    Return:
        train: numpy.ndarray
            Generated train.
    """

    if isinstance(rate, (float, str, unicode)):
        kwargs.update({'np': np})
        rate = eval("lambda t: {}".format(rate), kwargs)
    else:
        message = "Unknown rate type: {}".format(type(rate))
        raise TypeError(message)
    _ = kwargs  # Discard additional parameters.

    ref_time = 0.0
    times = []
    while ref_time < duration:
        # TODO check the following line.
        scale = 1.0 / rate(ref_time)
        # TODO remove the following line.
        # size = int((duration - ref_time) * rate(ref_time)) + 1
        size = 1
        intervals = np.random.exponential(scale=scale, size=size)
        times_ = ref_time + np.cumsum(intervals)
        times.append(times_[times_ < duration])
        ref_time = times_[-1]
    times = np.concatenate(times)
    train = Train(times)

    return train

# TODO implement generate_poisson_train.
# TODO implement generate_refractory_poisson_train.
# TODO adapt generate_train.


def generate_trains(nb_trains=3, **kwargs):
    """Generate trains.

    Parameters:
        nb_trains: integer (optional)
            Number of trains. The default value is 3.

    Return:
        trains: dictionary
            Generated trains.

    See also:
        circusort.io.generate_train
    """

    # TODO integrate a refractory period.

    trains = {
        k: generate_train(**kwargs)
        for k in range(0, nb_trains)
    }

    return trains


def save_train(path, train):
    """Save train to file.

    Parameters:
        path: string
            The path to the file in which to save the train.
        train: numpy.ndarray
            The train to save.
    """

    train.save(path)

    return


def save_trains(directory, trains, mode='default'):
    """Save trains to files.

    Parameters:
        directory: string
            Directory in which to save the trains.
        trains: dictionary
            Dictionary of trains.
        mode: string (optional)
            The mode to use to save the trains. Either 'default', 'by trains' or 'by cells'.
    """

    if not os.path.isdir(directory):
        os.makedirs(directory)

    if mode == 'default':

        # TODO complete.
        raise NotImplementedError()

    elif mode == 'by trains':

        train_directory = os.path.join(directory, "trains")
        if not os.path.isdir(train_directory):
            os.makedirs(train_directory)
        for k, train in trains.items():
            filename = "{}.h5".format(k)
            path = os.path.join(train_directory, filename)
            save_train(path, train)

    elif mode == 'by cells':

        for k, train in trains.items():
            cell_directory = os.path.join(directory, "cells", "{}".format(k))
            if not os.path.isdir(cell_directory):
                os.makedirs(cell_directory)
            filename = "train.h5".format(k)
            path = os.path.join(cell_directory, filename)
            save_train(path, train)

    else:

        message = "Unknown mode value: {}".format(mode)
        raise ValueError(message)

    return


def list_trains(directory):
    """List train paths contained in the specified directory.

    Parameter:
        directory: string
            Directory from which to list the trains.

    Return:
        paths: list
            List of train paths found in the specified directory.
    """

    if not os.path.isdir(directory):
        message = "No such train directory: {}".format(directory)
        raise OSError(message)

    filenames = os.listdir(directory)
    filenames.sort()
    paths = [os.path.join(directory, filename) for filename in filenames]

    return paths


def load_train(path, **kwargs):
    """Load train from path.

    Parameter:
        path: string
            Path from which to load the train.

    Return:
        train: numpy.array
            Train. An array of spike times.
    """

    _ = kwargs  # Discard additional keyword arguments.

    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    if os.path.isdir(path):
        path = os.path.join(path, "train.h5")
    if not os.path.isfile(path):
        message = "No such train file: {}".format(path)
        raise IOError(message)

    f = h5py.File(path, mode='r')
    times = f.get('times').value
    f.close()
    train = Train(times)

    return train


def load_trains(directory, **kwargs):
    """Load trains from files.

    Parameter:
        directory: string
            Directory from which to load the trains.

    Return:
        trains: dictionary
            Dictionary of trains.
    """

    _ = kwargs  # Discard additional keyword arguments.

    paths = list_trains(directory)

    trains = {
        k: load_train(path)
        for k, path in enumerate(paths)
    }

    return trains


def get_train(path=None, **kwargs):
    """Get train.

    Parameter:
        path: none | string (optional)
            The path to use to get the train. The default value is None.

    Return:
        train: numpy.ndarray
            The train to get.

    See also:
        circusort.io.generate_train
    """

    if isinstance(path, (str, unicode)):
        try:
            train = load_train(path, **kwargs)
        except IOError:
            train = generate_train(**kwargs)
    else:
        train = generate_train(**kwargs)

    return train
