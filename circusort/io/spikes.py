import h5py
import os

from circusort.obj.spikes import Spikes
from circusort.utils.path import normalize_path


def load_spikes(path, **kwargs):
    # TODO add docstring.

    _ = kwargs  # Discard additional keyword arguments.

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
            key: file_[key]
            for key in ['times', 'templates', 'amplitudes']
        }

    # Instantiate object.
    spikes = Spikes(**data)

    return spikes
