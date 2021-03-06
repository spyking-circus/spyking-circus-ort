import numpy as np
import os
import sys

from circusort.io.parameter import generate_parameters, load_parameters, get_parameters
from circusort.utils.path import normalize_path

if sys.version_info.major == 3:
    unicode = str  # Python 3 compatibility.


default_types = {
    'general': {
        'duration': 'float',
        'sampling_rate': 'float',
        'buffer_width': 'integer',
        'dtype': 'string',
        'name': 'string',
    },
    'probe': {
        'mode': 'string',
        'nb_rows': 'integer',
        'nb_columns': 'integer',
        'interelectrode_distance': 'float',
        'radius': 'float',
        'path': 'string',
    },
    'cells': {
        'mode': 'string',
        'nb_cells': 'integer',
    },
}

defaults = {
    'general': {
        'duration': 20.0,
        'sampling_rate': 20e+3,
        'buffer_width': 1024,
        'dtype': "int16",
        'name': "",
    },
    'probe': {
        'mode': "default",
        'nb_channels': 16,
        'nb_rows': 4,
        'nb_columns': 4,
        'interelectrode_distance': 30.0,
        'radius': 250.0,
        'path': "",
    },
    'cells': {
        'mode': "default",
        'nb_cells': 3,
    },
}


def generate_configuration(**kwargs):

    defaults_ = defaults.copy()
    defaults_.update(kwargs)
    configuration = generate_parameters(types=default_types, defaults=defaults_)

    return configuration


def list_configuration_names(path):

    # Normalize and check path.
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        message = "Configuration directory not found: {}".format(path)
        raise IOError(message)
    names = os.listdir(path)
    paths = [
        os.path.join(path, name)
        for name in names
    ]
    names = [
        name
        for name, path in zip(names, paths) if os.path.isdir(path)
    ]
    # TODO add path to the configuration directory if it contains a file named parameters.txt.
    # TODO take care that writes in the configuration directory may overwrite subdirectories.
    try:
        values = np.array([
            float(name)
            for name in names
        ])
        indices = np.argsort(values)
        names = [names[index] for index in indices]
    except ValueError:  # i.e. could not convert to float.
        pass

    return names


def list_configuration_paths(path):

    # Normalize and check path.
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        message = "Configuration directory not found: {}".format(path)
        raise IOError(message)
    names = list_configuration_names(path)
    paths = [
        os.path.join(path, name)
        for name in names
    ]

    return paths


def load_configuration(path):

    parameters = load_parameters(path, types=default_types)

    return parameters


def load_configurations(path):

    paths = list_configuration_paths(path)
    if paths:
        names = list_configuration_names(path)
        configurations = [
            load_configuration(path)
            for path in paths
        ]
        for configuration, name in zip(configurations, names):
            configuration['general']['name'] = name
    else:
        message = "No configuration found in directory: {}".format(path)
        raise IOError(message)

    return configurations


def get_configuration(path=None):

    parameters = get_parameters(path=path, defaults=defaults)

    return parameters


def get_configurations(path=None, **kwargs):

    if isinstance(path, (str, unicode)):
        path = normalize_path(path, **kwargs)
        paths = list_configuration_paths(path)
        if paths:
            names = list_configuration_names(path)
            configurations = [
                get_configuration(path=path)
                for path in paths
            ]
            for configuration, name in zip(configurations, names):
                configuration['general']['name'] = name
        else:
            raise NotImplementedError()  # TODO generate default configuration.
    else:
        raise NotImplementedError()  # TODO generate default configuration.

    return configurations
