import os

from collections import OrderedDict as odict

from .base import load_parameters


defaults = odict([
    ('template', odict([
        ('mode', ('string', 'default', 'Mode of generation.')),
        ('path', ('string', '', 'Path to the data file.')),
    ])),
    ('train', odict([
        ('mode', ('string', 'default', 'Mode of generation.')),
        ('rate', ('float', 10.0, 'Firing rate.')),
        ('path', ('string', '', 'Path to the data file.')),
    ])),
    ('position', odict([
        ('mode', ('string', 'default', 'Mode of generation.')),
        ('x', ('float', 0.0, 'x-coordinate.')),
        ('y', ('float', 0.0, 'y-coordinate.')),
        ('path', ('string', '', 'Path to the data file.')),
    ])),
])

required = {
    'template': ['mode'],
    'train': ['mode'],
    'position': ['mode'],
}


def generate_cell_parameters():
    """Generate cell parameters.

    Return:
        parameters: dictionary
            The parameters of the cell.
    """

    parameters = {}
    for section, options in required.iteritems():
        parameters[section] = {}
        for option in options:
            _, value, _ = defaults[section][option]
            parameters[section][option] = value

    return parameters


def load_cell_parameters(path):
    """Load parameters form file.

    Parameter:
        path: string
            The path to the file from which to load the parameters of the cell.

    Return:
        parameters: dictionary
            The parameters of the cell.
    """

    parameters = load_parameters(path, defaults=defaults)

    return parameters


def get_cell_parameters(path=None):
    """Get parameters from path.

    Parameter:
        path: none | string (optional)
            The path to use to get the parameters of the cell. The default value is None.

    Return:
        parameters: dictionary
            The parameters of the cell.
    """

    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    if path is None:
        parameters = generate_cell_parameters()
    elif not os.path.isfile(path):
        parameters = generate_cell_parameters()
    else:
        try:
            parameters = load_cell_parameters(path)
        except IOError:
            parameters = generate_cell_parameters()

    return parameters
