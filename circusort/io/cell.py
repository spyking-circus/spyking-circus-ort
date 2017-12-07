import os

from .template import list_templates, load_template
from .trains import list_trains, load_train


def load_cells(working_directory=None):
    """Load cells from the specified working directory.

    Parameter:
        working_directory: none | string (optional)
    """
    # TODO complete docstring.

    generation_directory = os.path.join(working_directory, "generation")

    template_directory = os.path.join(generation_directory, "template")
    template_paths = list_templates(template_directory)

    train_directory = os.path.join(generation_directory, "trains")
    train_paths = list_trains(train_directory)

    message = "Different number of templates and trains between {} and {}".format(template_directory, train_directory)
    assert len(template_paths) == len(train_paths), message

    cells = {}
    for template_path, train_path in zip(template_paths, train_paths):
        template = load_template(template_path)
        train = load_train(train_path)
        raise NotImplementedError()  # TODO complete.

    raise NotImplementedError()  # TODO complete.

    return cells
