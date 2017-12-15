from .base import load_parameters, get_parameters


default_cell_types = {
    'template': {
        'mode': 'string',
        'path': 'string',
    },
    'train': {
        'mode': 'string',
        'rate': 'string',
        'path': 'string',
    },
    'position': {
        'mode': 'string',
        'x': 'float',
        'y': 'float',
        'path': 'string',
    },
}

default_cells_types = {
    'general': {
        'nb_cells': 'integer',
    },
}


def load_cell_parameters(path):
    """Load cell parameters from file.

    Parameter:
        path: string
            The path to the file from which to load the parameters of the cell.

    Return:
        parameters: circusort.obj.Parameters
            The parameters of the cell.
    """

    parameters = load_parameters(path, types=default_cell_types)

    return parameters


def get_cell_parameters(path=None):
    """Get cell parameters from path.

    Parameter:
        path: none | string (optional)
            The path to use to get the parameters of the cell. The default value is None.

    Return:
        parameters: circusort.obj.Parameters
            The parameters of the cell.
    """

    parameters = get_parameters(path=path, types=default_cell_types)

    return parameters


def get_cells_parameters(path=None):
    """Get cells parameters from path.

    Parameter:
        path: none | string (optional)
            The path to use to get the parameters of the cells. The default value is None.

    Return:
        parameters: circusort.obj.Parameters
            The parameters of the cells.
    """

    parameters = get_parameters(path=path, types=default_cells_types, default_type='float')

    return parameters
