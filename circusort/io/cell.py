import os

from circusort.io.parameter.cell import load_cell_parameters, get_cell_parameters
from .template import generate_template, load_template, get_template
from .trains import generate_train, load_train, get_train
from .position import generate_position, load_position, get_position
from ..obj.cell import Cell


def generate_cell(**kwargs):
    # TODO add docstring.

    train = generate_train(**kwargs)
    position = generate_position(train=train, **kwargs)
    template = generate_template(position=position, **kwargs)
    # TODO generate amplitude.
    cell = Cell(template, train, position=position)

    return cell


def load_cell(directory):
    """Load cell from the specified directory.

    Parameter:
        directory: string
            The path to the directory from which to load the cell.

    Return:
        cell: circusort.obj.Cell
            The loaded cell.
    """

    # Get parameters.
    path = os.path.join(directory, "parameters.txt")
    parameters = load_cell_parameters(path)
    # Load template.
    path = os.path.join(directory, "template.h5")
    template = load_template(path)
    # Load train.
    path = os.path.join(directory, "train.h5")
    train = load_train(path)
    # Load position.
    path = os.path.join(directory, "position.h5")
    position = load_position(path)
    cell = Cell(template, train, position, parameters=parameters)

    return cell


def get_cell(directory=None, **kwargs):
    """Get cell to use during the generation.

    Parameter:
        directory: none | string (optional)
            The path to the directory in which to look for the cell.

    Return:
        cell: circusort.obj.Cell
            The cell to get.

    See also:
        circusort.io.get_template
        circusort.io.get_train
        circusort.io.get_position
    """

    parameters = get_cell_parameters(directory)

    parameters['train']['path'] = os.path.join(directory, 'train.h5')
    train_parameters = kwargs.copy()
    train_parameters.update(parameters['train'])
    train = get_train(**train_parameters)

    parameters['position']['path'] = os.path.join(directory, 'position.h5')
    position_parameters = kwargs.copy()
    position_parameters.update(parameters['position'])
    position = get_position(train=train, **position_parameters)

    parameters['template']['path'] = os.path.join(directory, 'template.h5')
    template_parameters = kwargs.copy()
    template_parameters.update(parameters['template'])
    template = get_template(position=position, **template_parameters)

    cell = Cell(template, train, position=position, parameters=parameters)

    return cell
