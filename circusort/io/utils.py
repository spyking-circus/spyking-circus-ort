import os



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
