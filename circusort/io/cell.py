import os

from circusort import io


def load_cells(working_directory=None):
    """Load cells from the specified working directory.

    Parameter:
        working_directory: none | string (optional)
    """
    # TODO complete docstring.

    generation_directory = os.path.join(working_directory, "generation")

    template_directory = os.path.join(generation_directory, "template")
    template_paths = io.list_templates(template_directory)

    train_directory = os.path.join(generation_directory, "trains")
    train_paths = io.list_trains(train_directory)

    message = "Different number of templates and trains between {} and {}".format(template_directory, train_directory)
    assert len(template_paths) == len(train_paths), message

    cells = {}
    for template_path, train_path in zip(template_paths, train_paths):
        template = io.load_template(template_path)
        train = io.load_train(train_path)
        raise NotImplementedError()  # TODO complete.

    raise NotImplementedError()  # TODO complete.

    return cells
