from circusort.io.parameter import load_parameters, get_parameters


default_types = {
    'general': {
        'duration': 'float',
        'sampling_rate': 'float',
        'buffer_width': 'integer',
        'dtype': 'string',
    },
    'probe': {
        'mode': 'string',
        'nb_rows': 'integer',
        'nb_columns': 'float',
        'interelectrode_distance': 'float',
    },
    'cells': {
        'mode': 'string',
    },
}


def load_data_parameters(path):
    """Load data parameters from file.

    Parameter:
        path: string
            The path to the file from which to load the parameters of the data.

    Return:
        parameters: circusort.obj.Parameters
            The parameters of the data.
    """

    parameters = load_parameters(path, types=default_types)

    return parameters


def get_data_parameters(path=None):
    """Get data parameters from path.

    Parameter:
        path: none | string (optional)
            The path to use to get the parameters of the data. The default value is None.

    Return:
        parameters: circusort.obj.Parameters
            The parameters of the data.
    """

    parameters = get_parameters(path=path, types=default_types)

    return parameters
