import os

from circusort.io.template import load_template
from circusort.io.templates import list_templates
from circusort.io.trains import list_trains, load_train
from circusort.io.position import list_positions, load_position
from circusort.io.cell import generate_cell, get_cell
from circusort.io.parameter.cells import get_cells_parameters
from circusort.obj.cell import Cell
from circusort.obj.cells import Cells
from circusort.utils.path import normalize_path


def generate_cells(nb_cells=3, **kwargs):
    # TODO add docstring.

    cells = {
        k: generate_cell(**kwargs)
        for k in range(0, nb_cells)
    }
    cells = Cells(cells)

    return cells


def save_cells(directory, cells, **kwargs):
    """Save cells to files.

    Parameters:
        directory: string
            The path to the directory in which to save the cells.
        cells: circusort.obj.Cells
            The cells to save.
    """

    cells.save(directory, **kwargs)

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


def load_cells(path=None, mode='default', **kwargs):
    """Load cells from the specified path.

    Parameter:
        path: none | string (optional)
            The path to the directory from which to load the cells. The default value is None.
        mode: string (optional)
            The mode to use to load the cells. Either 'default' or 'by cells. The default value is 'default'.

    Return:
        cells: dictionary
            Dictionary of loaded cells.
    """

    if mode == 'by elements':

        template_directory = os.path.join(path, "templates")
        if not os.path.isdir(template_directory):
            message = "No such templates directory: {}".format(template_directory)
            raise OSError(message)
        template_paths = list_templates(template_directory)

        train_directory = os.path.join(path, "trains")
        if not os.path.isdir(train_directory):
            message = "No such trains directory: {}".format(train_directory)
            raise OSError(message)
        train_paths = list_trains(train_directory)

        position_directory = os.path.join(path, "positions")
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
        cells = Cells(cells)

    elif mode in ['default', 'by cells']:

        path = normalize_path(path, **kwargs)
        if path[-6:] != "/cells":
            path = os.path.join(path, "cells")
        if not os.path.isdir(path):
            message = "No such cells directory: {}".format(path)
            raise OSError(message)
        parameters = get_cells_parameters(path)

        kwargs.update(parameters['general'])
        cell_directories = list_cells(path)
        cells = {
            k: get_cell(directory=cell_directory, **kwargs)
            for k, cell_directory in enumerate(cell_directories)
        }
        cells = Cells(cells, parameters=parameters)

    else:

        message = "Unknown mode value: {}".format(mode)
        raise ValueError(message)

    return cells


def get_cells(path=None, **kwargs):
    """Get cells to use during the generation.

    Parameter:
        path: none | string (optional)
            The path to the directory in which to look for the cells.

    Return:
        cells: dictionary
            The cells to get.

    See also:
        circusort.io.generate_cells
        circusort.io.get_cell
    """

    if isinstance(path, (str, unicode)):
        path = normalize_path(path, **kwargs)
        if path[-6:] != "/cells":
            path = os.path.join(path, "cells")
        if os.path.isdir(path):
            try:
                cells = load_cells(path, **kwargs)
            except OSError:
                cells = generate_cells(**kwargs)
        else:
            cells = generate_cells(**kwargs)
    else:
        cells = generate_cells(**kwargs)

    return cells
