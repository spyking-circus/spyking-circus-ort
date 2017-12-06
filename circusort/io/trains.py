import h5py
import numpy as np
import os


def generate_trains(nb_trains=3, duration=60.0, rate=1.0):
    """Generate trains.

    Parameters:
        nb_trains: integer (optional)
            Number of trains. The default value is 3.
        duration: float (optional)
            Train duration [s]. The default value is 60.0.
        rate: float (optional)
            Spike rate [Hz]. The default value is 1.0.
    """

    # TODO integrate a refractory period.

    trains = {}

    for k in range(0, nb_trains):
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
        trains[k] = train

    return trains


def save_trains(directory, trains):
    """Save trains to files.

    Parameters:
        directory: string
            Directory in which to save the trains.
        trains: dictionary
            Dictionary of trains.
    """

    if not os.path.isdir(directory):
        os.makedirs(directory)

    for k, times in trains.iteritems():
        filename = "{}.h5".format(k)
        path = os.path.join(directory, filename)
        f = h5py.File(path, mode='w')
        f.create_dataset('times', shape=times.shape, dtype=times.dtype, data=times)
        f.close()

    return


def load_trains(directory):
    """Load trains from files.

    Parameter:
        directory: string
            Directory from which to load the trains.

    Return:
        trains: dictionary
            Dictionary of trains.
    """

    if not os.path.isdir(directory):
        message = "No such train directory: {}".format(directory)
        raise OSError(message)

    filenames = os.listdir(directory)
    filenames.sort()

    trains = {}

    for k, filename in enumerate(filenames):
        path = os.path.join(directory, filename)
        f = h5py.File(path, mode='r')
        times = f['times']
        f.close()
        train = times
        trains[k] = train

    return trains
