import ConfigParser
import os

from collections import OrderedDict

from circusort.obj.parameter import Parameters


default_parameters = OrderedDict([
    ('generation', OrderedDict([
        ('duration', ('float', 60.0, "s  # Duration of the signal.")),
        ('sampling rate', ('float', 20000.0, "Hz  # Number of sampling times per second.")),
        ('buffer width', ('integer', 1024, "Number of sampling times per buffer.")),
        ('dtype', ('string', 'int16', "Data type.")),
    ])),
])


def generate_parameters(types=None):
    """Generate the parameters to use during the generation."""
    # TODO complete docstring.

    parameters = [
        (section, OrderedDict())
        for section in types.iterkeys()
    ]
    parameters = Parameters(parameters)

    return parameters


def save_parameters(path, parameters):
    """Save the parameters to use during the generation.

    Parameters:
        path: string
            The path to the file in which to save the parameters.
        parameters: dictionary
            The parameters to save.
    """

    # TODO add a keyword argument for comments.

    # Complete path (if necessary).
    if path[-4:] != ".txt":
        path = os.path.join(path, "parameters.txt")

    # Make directories (if necessary).
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Prepare lines to be saved.
    lines = []
    for section in parameters:
        line = "[{}]\n".format(section)
        lines.append(line)
        for option in parameters[section]:
            if option != 'current directory':
                value = parameters[section][option]
                line = "{} = {}\n".format(option, value)
                lines.append(line)
        line = "\n"
        lines.append(line)
    if not lines:
        line = "\n"
        lines.append(line)

    # Open probe file.
    file_ = open(path, mode='w')
    # Write lines to save.
    file_.writelines(lines)
    # Close probe file.
    file_.close()

    return


def load_parameters(path, types=None):
    """Load parameters from a file saved on disk.

    Parameter:
        path: string
            The path to the file from which to load the parameters.
        types: none | dictionary
            The default types of the parameters. The default value is None.

    Return:
        parameters: dictionary
            The loaded parameters.
    """

    # Normalize and check path.
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    if os.path.isdir(path):
        path = os.path.join(path, "parameters.txt")
    if not os.path.isfile(path):
        message = "No such parameters file: {}".format(path)
        raise IOError(message)

    # Define working directory.
    current_directory = os.path.dirname(path)

    # Read parameters file from disk.
    parser = ConfigParser.ConfigParser()
    parser.read(path)

    # Remove comments at the end of each line.
    for section_name in parser.sections():
        for option_name in parser.options(section_name):
            value = parser.get(section_name, option_name)  # get value
            words = value.split('#')  # split value and end line comment
            word = words[0]  # keep value
            value = word.strip()  # remove leading and trailing characters
            parser.set(section_name, option_name, value)  # set value

    # From ConfigParser to dictionary.
    parameters = []
    for section_name in parser.sections():
        section = [('current directory', current_directory)]
        for option_name in parser.options(section_name):
            if types is None:
                type_ = 'string'
            else:
                type_ = types[section_name][option_name]
            if type_ == 'boolean':
                value = parser.getboolean(section_name, option_name)
            elif type_ == 'integer':
                value = parser.getint(section_name, option_name)
            elif type_ == 'float':
                value = parser.getfloat(section_name, option_name)
            elif type_ == 'string':
                value = parser.get(section_name, option_name)
            else:
                message = "Unknown type {}".format(type_)
                raise ValueError(message)
            section.append((option_name, value))
        parameters.append((section_name, section))

    # Instantiate object.
    parameters = Parameters(parameters)

    return parameters


def get_parameters(path=None, types=None):
    """Get parameters from path.

    Parameter:
        path: none | string
            The path to the file from which to load the parameters. The default value is None.
        types: none | dictionary
            The default types of the parameters. The default value is None.

    Return:
        parameters: dictionary
            The loaded parameters.
    """

    if isinstance(path, (str, unicode)):
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        if os.path.isdir(path):
            path = os.path.join(path, "parameters.txt")
        if os.path.isfile(path):
            try:
                parameters = load_parameters(path, types=types)
            except (IOError, ValueError):
                parameters = generate_parameters(types=types)
        else:
            parameters = generate_parameters(types=types)
    else:
        parameters = generate_parameters(types=types)

    return parameters
