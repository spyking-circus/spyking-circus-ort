import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
import numpy as np
import os

from circusort.utils.path import normalize_path


def _get_filename(name):

    if name is None:
        filename = "time_measurements.h5"
    else:
        filename = "time_measurements_" + name + ".h5"

    return filename


def save_time_measurements(path, measurements, name=None):
    # TODO add docstring.

    path = normalize_path(path)
    _, extension = os.path.splitext(path)
    if extension == '':
        filename = _get_filename(name)
        path = os.path.join(path, filename)
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    with h5py.File(path, mode='w', swmr=True) as file_:
        for label in measurements:
            times = np.array(measurements[label])
            file_.create_dataset(name=label, data=times)

    return


def load_time_measurements(path, name=None):
    # TODO add docstring.

    path = normalize_path(path)
    if os.path.isdir(path):
        filename = _get_filename(name)
        path = os.path.join(path, filename)
    if not os.path.isfile(path):
        message = "Time measurements file does not exist: {}".format(path)
        raise OSError(message)

    with h5py.File(path, mode='r', swmr=True) as file_:
        measurements = {
            dataset_name: file_[dataset_name].value
            for dataset_name in file_.keys()
        }

    return measurements
