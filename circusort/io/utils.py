import os
import tempfile
try:
    # Python 2 compatibility.
    from cPickle import dump as pickle_dump
    from cPickle import load as pickle_load
except ImportError:  # i.e. ModuleNotFoundError
    # Python 3 compatibility.
    from _pickle import dump as pickle_dump
    from _pickle import load as pickle_load


def decompose_path(path):

    root, extension = os.path.splitext(path)
    directory, basename = os.path.split(root)

    return directory, basename, extension


def get_local_directory(path):

    directory, basename, _ = decompose_path(path)
    path = os.path.join(directory, basename)

    return path


def get_header_path(path):

    directory, basename, _ = decompose_path(path)
    filename = basename + ".header"
    path = os.path.join(directory, filename)

    return path


def get_spatial_configuration_path(path):

    directory = get_local_directory(path)
    filename = "spatial_configuration.png"
    path = os.path.join(directory, filename)

    return path


def get_temporal_configuration_path(path):

    directory = get_local_directory(path)
    filename = "temporal_configuration.png"
    path = os.path.join(directory, filename)

    return path


def get_waveforms_path(path):

    directory = get_local_directory(path)
    filename = "waveforms.png"
    path = os.path.join(directory, filename)

    return path


def make_local_directory(path):

    directory = get_local_directory(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    return


def generate_fake_probe(nb_channels, radius=10, prb_file=None):

    res = '''
total_nb_channels = %d
radius            = %d
channel_groups    = {}

def get_geometry(channels):
    res = {}
    for count, c in enumerate(channels):
        res[c] = [count, count]
    return res

channel_groups[0]             = {}
channel_groups[0]["channels"] = range(total_nb_channels)
channel_groups[0]["geometry"] = get_geometry(range(total_nb_channels))
channel_groups[0]["graph"]    = []''' % (nb_channels, radius)

    if prb_file is None:
        tmp_file = tempfile.NamedTemporaryFile()
        prb_file = os.path.join(tempfile.gettempdir(), os.path.basename(tmp_file.name)) + ".prb"
        tmp_file.close()
    else:
        prb_file = os.path.abspath(prb_file)
    file = open(prb_file, 'w')
    file.write(res)
    file.close()

    return prb_file


def save_pickle(filename, data):

    file = open(filename + '.pck', 'w')
    pickle_dump(data, file)
    file.close()

    return


def load_pickle(filename):

    file = open(filename + '.pck', 'r')
    res = pickle_load(file)
    file.close()

    return res


def append_hdf5(dataset, data):
    """Append 1D-array to a HDF5 dataset.

    Arguments:
    dataset: ?
        HDF5 dataset.
    data: numpy.ndarray
        1D-array.
    """

    old_size = len(dataset)
    new_size = old_size + len(data)
    if len(dataset.shape) == 1:
        new_shape = (new_size,)
    else:
        new_shape = (new_size, dataset.shape[1])
    dataset.resize(new_shape)
    dataset[old_size:new_size] = data

    return
