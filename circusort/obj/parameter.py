import os

from collections import OrderedDict


def to_ordered_dictionary(list_):
    # TODO add docstring.

    dict_ = OrderedDict()
    for (section_name, section) in list_:
        dict_[section_name] = {}
        for (option_name, value) in section:
            dict_[section_name][option_name] = value

    return dict_


class Parameters(object):
    # TODO add docstring.

    def __init__(self, parameters=None, mode='default'):
        # TODO add docstring.

        if mode == 'default':
            if parameters is None:
                parameters = []
            self.parameters = to_ordered_dictionary(parameters)
        else:
            message = "Unknown mode value: {}".format(mode)
            raise ValueError(message)

    def __getitem__(self, key):
        # TODO add docstring.

        value = self.parameters.get(key, OrderedDict())

        return value

    def __setitem__(self, key, value):
        # TODO add docstring.

        self.parameters.__setitem__(key, value)

        return

    def __contains__(self, key):
        # TODO add docstring.

        is_contained = key in self.parameters

        return is_contained

    def __iter__(self):
        # TODO add docstring.

        iterator = self.parameters.iterkeys()

        return iterator

    def add(self, section, option, value):
        # TODO add docstring.

        self.parameters[section][option] = value

        return

    def update(self, parameters):
        # TODO add docstring.

        self.parameters.update(parameters.parameters)

        return

    def save(self, path):
        # TODO add docstring.

        # Normalize path.
        path = os.path.expanduser(path)
        path = os.path.abspath(path)

        # Make directory (if necessary).
        directory = os.path.dirname(path)
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # Prepare lines to be saved.
        lines = []
        for section in self.parameters.iterkeys():
            line = "[{}]\n".format(section)
            lines.append(line)
            for option in self.parameters[section].iterkeys():
                value = self.parameters[section][option]
                line = "{} = {}\n".format(option, value)
                lines.append(line)
            line = "\n"
            lines.append(line)

        # Open probe file.
        file_ = open(path, mode='w')
        # Write lines to save.
        file_.writelines(lines)
        # Close probe file.
        file_.close()

        return
