import h5py
import numpy as np
import os


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

    _ = kwargs

    scale = 1.0 / rate
    time = 0.0
    train = []
    while time < duration:
        size = int((duration - time) * rate) + 1
        intervals = np.random.exponential(scale=scale, size=size)
        times = time + np.cumsum(intervals)
        train.append(times[times < duration])
        time = times[-1]
    train = np.concatenate(train)

    return train


def generate_trains(nb_trains=3, duration=60.0, rate=1.0):
    """Generate trains.

    Parameters:
        nb_trains: integer (optional)
            Number of trains. The default value is 3.
        duration: float (optional)
            Train duration [s]. The default value is 60.0.
        rate: float (optional)
            Spike rate [Hz]. The default value is 1.0.

    Return:
        trains: dictionary
            Generated trains.
    """

    # TODO integrate a refractory period.

    trains = {}

    for k in range(0, nb_trains):
        train = generate_train(duration=duration, rate=rate)
        trains[k] = train

    return trains


def save_train(path, train):
    """Save train to file.

    Parameters:
        path: string
            The path to the file in which to save the train.
        train: numpy.ndarray
            The train to save.
    """

    times = train

    f = h5py.File(path, mode='w')
    f.create_dataset('times', shape=times.shape, dtype=times.dtype, data=times)
    f.close()

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
        for k, train in trains.iteritems():
            filename = "{}.h5".format(k)
            path = os.path.join(train_directory, filename)
            save_train(path, train)

    elif mode == 'by cells':

        for k, train in trains.iteritems():
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


def load_train(path):
    """Load train from path.

    Parameter:
        path: string
            Path from which to load the train.

    Return:
        train: numpy.array
            Train. An array of spike times.
    """

    f = h5py.File(path, mode='r')
    times = f.get('times').value
    f.close()
    train = times

    return train


def load_trains(directory):
    """Load trains from files.

    Parameter:
        directory: string
            Directory from which to load the trains.

    Return:
        trains: dictionary
            Dictionary of trains.
    """

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
        circusort.io.generate_train (for additional parameters)
    """

    if path is None:
        template = generate_train(**kwargs)
    elif not os.path.isfile(path):
        template = generate_train(**kwargs)
    else:
        try:
            template = load_train(path)
        except OSError:
            template = generate_train(**kwargs)

    return template
