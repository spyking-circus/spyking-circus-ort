import os
import tempfile
import numpy
import scipy.sparse
import cPickle

def decompose_path(path):
    root, extension = os.path.splitext(path)
    directory, basename = os.path.split(root)
    return directory, basename, extension

def get_local_directory(path):
    directory, basename, _ = decompose_path(path)
    path = os.path.join(directory, basename)
    return path


def get_header_path(path):
    '''TODO add docstring...'''
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
channel_groups[0]["graph"]    = []''' %(nb_channels, radius)

    if prb_file is None:
        tmp_file  = tempfile.NamedTemporaryFile()
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
    cPickle.dump(data, file)
    file.close()

def load_pickle(filename):
    file = open(filename + '.pck', 'r')
    res  = cPickle.load(file)
    file.close()
    return res
    