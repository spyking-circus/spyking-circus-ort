import os

from .template import list_templates, load_template
from .trains import list_trains, load_train
from ..obj.cell import Cell


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
    directories = [os.path.join(directory, name) for name in names]

    return directories


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
        template_paths = list_templates(template_directory)

        train_directory = os.path.join(directory, "trains")
        train_paths = list_trains(train_directory)

        string = "Different number of templates and trains between {} and {}"
        message = string.format(template_directory, train_directory)
        assert len(template_paths) == len(train_paths), message

        cells = {}
        for k, (template_path, train_path) in enumerate(zip(template_paths, train_paths)):
            template = load_template(template_path)
            train = load_train(train_path)
            cell = Cell(template, train)
            cells[k] = cell

    elif mode == 'by cells':

        cells_directory = os.path.join(directory, "cells")
        cell_directories = list_cells(cells_directory)
        cells = {}
        for k, cell_directory in enumerate(cell_directories):
            template_path = os.path.join(cell_directory, "template.h5")
            train_path = os.path.join(cell_directory, "train.h5")
            template = load_template(template_path)
            train = load_train(train_path)
            cell = Cell(template, train)
            cells[k] = cell

    else:

        message = "Unknown mode value: {}".format(mode)
        raise ValueError(message)

    return cells
