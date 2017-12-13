import os

from .parameter import load_cell_parameters, get_cell_parameters
from .template import generate_template, list_templates, load_template, get_template
from .trains import generate_train, list_trains, load_train, get_train
from .position import generate_position, list_positions, load_position, get_position
from ..obj.cell import Cell


def generate_cells(nb_cells=3):
    # TODO add docstring.

    cells = {}
    for k in range(0, nb_cells):
        template = generate_template()
        train = generate_train()
        position = generate_position()
        cell = Cell(template, train, position)
        cells[k] = cell

    return cells


def save_cells(directory, cells, mode='default'):
    """Save cells to files.

    Parameters:
        directory: string
            Directory in which to save the cells.
        cells: dictionary
            Dictionary of cells to save.
        mode: string (optional)
            The mode to use to save the cells. Either 'default' or 'by cells'. The default value is 'default'.
    """

    if mode == 'default':

        raise NotImplementedError()  # TODO complete.

    elif mode == 'by cells':

        cells_directory = os.path.join(directory, "cells")
        for k, cell in cells.iteritems():
            cell_directory = os.path.join(cells_directory, "{}".format(k))
            cell.save(cell_directory)

    else:

        message = "Unknown mode value: {}".format(mode)
        raise ValueError(message)

    return


def list_cells(directory):
    """List cell directories contained in the specified directory.

    Parameter:
        directory: string
            The directory from which to list the cells.

    Return:
        directories: list
            List of cell directories found in the specified directory.
    """

    if not os.path.isdir(directory):
        message = "No such cells directory: {}".format(directory)
        raise OSError(message)

    names = os.listdir(directory)
    names.sort()
    paths = [os.path.join(directory, name) for name in names]
    directories = [path for path in paths if os.path.isdir(path)]

    return directories


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


def load_cells(directory=None, mode='default'):
    """Load cells from the specified directory.

    Parameter:
        directory: none | string (optional)
            The path to the directory from which to load the cells. The default value is None.
        mode: string (optional)
            The mode to use to load the cells. Either 'default' or 'by cells. The default value is 'default'.

    Return:
        cells: dictionary
            Dictionary of loaded cells.
    """

    if mode == 'default':

        template_directory = os.path.join(directory, "templates")
        if not os.path.isdir(template_directory):
            message = "No such templates directory: {}".format(template_directory)
            raise OSError(message)
        template_paths = list_templates(template_directory)

        train_directory = os.path.join(directory, "trains")
        if not os.path.isdir(train_directory):
            message = "No such trains directory: {}".format(train_directory)
            raise OSError(message)
        train_paths = list_trains(train_directory)

        position_directory = os.path.join(directory, "positions")
        if not os.path.isdir(position_directory):
            message = "No such positions directory: {}".format(position_directory)
            raise OSError(message)
        position_paths = list_positions(position_directory)

        string = "Different number of templates and trains between {} and {}"
        message = string.format(template_directory, train_directory)
        assert len(template_paths) == len(train_paths), message

        string = "Different number of templates and positions between {} and {}"
        message = string.format(template_directory, position_directory)
        assert len(template_paths) == len(position_paths), message

        nb_cells = len(template_paths)
        cells = {}
        for k in range(0, nb_cells):
            template = load_template(template_paths[k])
            train = load_train(train_paths[k])
            position = load_position(position_paths[k])
            cell = Cell(template, train, position)
            cells[k] = cell

    elif mode == 'by cells':

        cells_directory = os.path.join(directory, "cells")
        if not os.path.isdir(cells_directory):
            message = "No such cells directory: {}".format(directory)
            raise OSError(message)
        cell_directories = list_cells(cells_directory)
        cells = {
            k: load_cell(cell_directory)
            for k, cell_directory in enumerate(cell_directories)
        }

    else:

        message = "Unknown mode value: {}".format(mode)
        raise ValueError(message)

    return cells


def get_cell(directory=None, **kwargs):
    """Get cell to use during the generation.

    Parameter:
        directory: none | string (optional)
            The path to the directory in which to look for the cell.

    Return:
        cell: circusort.obj.Cell
            Cell.
    """

    path = os.path.join(directory, "parameters.txt")
    parameters = get_cell_parameters(path)

    template_parameters = kwargs.copy()
    template_parameters.update(parameters['template'])
    template = get_template(**template_parameters)

    train_parameters = kwargs.copy()
    train_parameters.update(parameters['train'])
    train = get_train(**train_parameters)

    position_parameters = kwargs.copy()
    position_parameters.update(parameters['position'])
    position = get_position(**position_parameters)

    cell = Cell(template, train, position, parameters=parameters)

    return cell


def get_cells(directory=None, **kwargs):
    """Get cells to use during the generation.

    Parameter:
        directory: none | string (optional)
            The path to the directory in which to look for the cells.

    Return:
        cells: dictionary
            Cells.
    """

    if directory is None:
        cells = generate_cells()
    else:
        # TODO check if there is a parameter file.
        # TODO load this parameter file.
        # parameters_path = os.path.join(directory, "parameters.txt")
        # parameters = get_parameters(parameters_path)
        cells_directory = os.path.join(directory, "cells")
        # Check if the cells directory exists.
        if os.path.isdir(cells_directory):
            # TODO check if there is a parameter file.
            # TODO load this parameter file.
            # parameters_path = os.path.join(cells_directory, "parameters.txt")
            # parameters = get_parameters(parameters_path)
            # List the cell directories.
            cell_directories = list_cells(cells_directory)
            if not cell_directories:
                # TODO generate cells with the parameters from the directory.
                pass
            else:
                cells = {
                    k: get_cell(directory=cell_directory, **kwargs)
                    for k, cell_directory in enumerate(cell_directories)
                }
        else:
            # TODO generate cells with the parameters from the directory.
            # Raise an error.
            message = "No such cells directory: {}".format(cells_directory)
            raise OSError(message)

    return cells
