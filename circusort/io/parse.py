import argparse
import os


def parse_parameters():
    # TODO add docstring.

    # 1. Parse command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--wd', dest='working_directory', action='store')
    # TODO add arguments?
    args = parser.parse_args()
    input_params = vars(args)

    # 2. Define/check working directory (if necessary)
    working_directory = input_params['working_directory']
    if working_directory is None:
        input_params['working_directory'] = os.getcwd()
    else:
        working_directory = os.path.expanduser(working_directory)
        working_directory = os.path.abspath(working_directory)
        if not os.path.isdir(working_directory):
            os.makedirs(working_directory)
        input_params['working_directory'] = working_directory

    # 2. Parse the local parameter file
    working_directory = input_params['working_directory']
    filename = get_parameter_filename(working_directory)
    if filename is None:
        local_params = {}
    else:
        local_params = parse_parameter_file(filename)

    # 3. Parse the global parameter file
    parameter_directory = os.path.join("~", ".spyking-circus-ort")
    parameter_directory = os.path.expanduser(parameter_directory)
    filename = get_parameter_filename(parameter_directory)
    if filename is None:
        global_params = {}
    else:
        global_params = parse_parameter_file(filename)

    # 4. Fusion input, local and global parameters.
    params = {}
    params.update(global_params)
    params.update(local_params)
    params.update(input_params)

    return params


def get_parameter_filename(path):
    # TODO add docstring

    parameter_filenames = [
        "config.txt",
        "config.params",
    ]

    if not os.path.exists(path):
        os.makedirs(path)
    
    names = os.listdir(path)

    parameter_filename = None
    for filename in parameter_filenames:
        if filename in names:
            parameter_filename = filename
            break

    return parameter_filename


def parse_parameter_file(path):
    # TODO add docstring.

    params = {}
    # TODO complete.

    return params
