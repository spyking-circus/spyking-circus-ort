import os


def normalize_path(path, current_directory=None, **kwargs):
    # TODO add docstring.

    _ = kwargs  # Discard additional keyword arguments.

    if current_directory is None:
        current_directory = os.getcwd()

    path = os.path.expanduser(path)
    if path in [".", "./"]:
        path = current_directory
    elif path[:2] == "./":
        path = os.path.join(current_directory, path[2:])
    else:
        path = os.path.abspath(path)

    return path
