import h5py
import numpy as np
import os

from circusort.obj.spikes import Spikes
from circusort.utils.path import normalize_path


def load_spikes(*args, **kwargs):
    # TODO add docstring.

    mode = kwargs.pop('mode', None)
    if mode is None:
        if len(args) == 1:
            mode = 'hdf5'
        elif len(args) == 3:
            mode = 'raw'
        else:
            string = "load_spikes takes exactly 1 or 3 arguments (in 'raw' or 'hdf5' mode) ({} given)"
            message = string.format(len(args))
            raise TypeError(message)

    if mode == 'raw':

        if len(args) != 3:
            message = "load_spikes takes exactly 3 argument (in 'hdf5' mode) ({} given)".format(len(args))
            raise TypeError(message)
        times_path, templates_path, amplitudes_path = args

        # Normalize paths.
        times_path = normalize_path(times_path)
        if os.path.isdir(times_path):
            os.path.join(times_path, "spikes_times.raw")
        if not os.path.isfile(times_path):
            message = "File does not exist: {}".format(times_path)
            raise OSError(message)
        templates_path = normalize_path(templates_path)
        if os.path.isdir(templates_path):
            os.path.join(templates_path, "templates.h5")
        if not os.path.isfile(templates_path):
            message = "File does not exist: {}".format(templates_path)
            raise OSError(message)
        amplitudes_path = normalize_path(amplitudes_path)
        if os.path.isdir(amplitudes_path):
            os.path.join(amplitudes_path, "amplitudes.h5")
        if not os.path.isfile(amplitudes_path):
            message = "File does not exist: {}".format(amplitudes_path)
            raise OSError(message)

        data = {
            'times': np.fromfile(times_path, dtype=np.int32),
            'templates': np.fromfile(templates_path, dtype=np.int32),
            'amplitudes': np.fromfile(amplitudes_path, dtype=np.float32),
        }

    elif mode == 'hdf5':

        if len(args) != 1:
            message = "load_spikes takes exactly 1 argument (in 'raw' mode) ({} given)".format(len(args))
            raise TypeError(message)
        path, = args

        # Normalize path.
        path = normalize_path(path, **kwargs)
        if os.path.isdir(path):
            os.path.join(path, "spikes.h5")
        if not os.path.isfile(path):
            message = "File does not exist: {}".format(path)
            raise OSError(message)

        # Read data from HDF5 file.
        with h5py.File(path, mode='r', swmr=True) as file_:
            data = {
                key: file_[key].value
                for key in ['times', 'templates', 'amplitudes']
            }

    else:

        message = "Unknown mode value: {}".format(mode)
        raise ValueError(message)

    # Instantiate object.
    spikes = Spikes(**data)

    return spikes
