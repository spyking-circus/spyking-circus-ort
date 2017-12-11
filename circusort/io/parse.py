import argparse
import ConfigParser as configparser
import os


def parse_parameters():
    # TODO add docstring.

    # 1. Parse command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('--wd', dest='working_directory', action='store')
    # TODO add arguments?
    args = parser.parse_args()
    input_params = vars(args)

    # 2. Define/check working directory (if necessary).
    working_directory = input_params['working_directory']
    if working_directory is None:
        input_params['working_directory'] = os.getcwd()
    else:
        working_directory = os.path.expanduser(working_directory)
        working_directory = os.path.abspath(working_directory)
        if not os.path.isdir(working_directory):
            os.makedirs(working_directory)
        input_params['working_directory'] = working_directory

    # 2. Parse the local parameter file.
    working_directory = input_params['working_directory']
    path = find_parameters_path(working_directory)
    local_params = parse_parameters_file(path) if path is not None else {}

    # 3. Parse the global parameter file.
    parameter_directory = os.path.join("~", ".spyking-circus-ort")
    parameter_directory = os.path.expanduser(parameter_directory)
    path = find_parameters_path(parameter_directory)
    global_params = parse_parameters_file(path) if path is not None else {}

    # 4. Fusion input, local and global parameters.
    params = {}
    params.update(global_params)
    params.update(local_params)
    params.update(input_params)

    return params


def find_parameters_path(directory):
    # TODO add docstring

    parameters_filenames = [
        "parameters.txt",
    ]

    if not os.path.exists(directory):
        path = None
    else:
        names = os.listdir(directory)
        path = None
        for filename in parameters_filenames:
            if filename in names:
                path = os.path.join(directory, filename)
                break

    return path


def parse_parameters_file(path):
    """Parse a parameters file from disk.

    Parameter:
        path: string
            The path to the file from which to parse the parameters.
    """

    # Read parameters file from disk.
    parser = configparser.ConfigParser()
    parser.read(path)

    # Remove comments at the end of each line.
    for section in parser.sections():
        for option in parser.options(section):
            value = parser.get(section, option)  # get value
            words = value.split('#')  # split value and end line comment
            word = words[0]  # keep value
            value = word.strip()  # remove leading and trailing characters
            parser.set(section, option, value)  # set value


    # Data types of the configuration values.
    types = {
        'generation': {
            'duration': 'float',
        },
    }

    # From ConfigParser to dictionary.
    params = {}
    for section in parser.sections():
        params[section] = {}
        for option in parser.options(section):
            type_ = types[section][option]
            if type_ == 'boolean':
                value = parser.getboolean(section, option)
            elif type_ == 'integer':
                value = parser.getint(section, option)
            elif type_ == 'float':
                value = parser.getfloat(section, option)
            elif type_ == 'string':
                value = parser.get(section, option)
            else:
                message = "Unknown type {}".format(type_)
                raise ValueError(message)
            params[section][option] = value

    return params
