from .base import load_parameters, get_parameters


default_types = {
    'general': {
        'nb_cells': 'integer',
    },
}


def load_cells_parameters(path):
    """Load cells parameters from file.

    Parameter:
        path: string
            The path to the file from which to load the parameters of the cells.

    Return:
        parameters: circusort.obj.Parameters
            The parameters of the cell.
    """

    parameters = load_parameters(path, types=default_types)

    return parameters


def get_cells_parameters(path=None, **kwargs):
    """Get cells parameters from path.

    Parameter:
        path: none | string (optional)
            The path to use to get the parameters of the cells. The default value is None.

    Return:
        parameters: circusort.obj.Parameters
            The parameters of the cells.
    """

    parameters = get_parameters(path=path, types=default_types, default_type='float')

    return parameters
