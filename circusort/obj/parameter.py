import os

from collections import OrderedDict

from circusort.utils.path import normalize_path


def to_ordered_dictionary(list_):

    dict_ = OrderedDict()
    for (section_name, section) in list_:
        dict_[section_name] = {}
        for (option_name, value) in section:
            dict_[section_name][option_name] = value

    return dict_


class Parameters(object):

    def __init__(self, parameters=None, mode='default'):

        if mode == 'default':
            if parameters is None:
                parameters = []
            self.parameters = to_ordered_dictionary(parameters)
        else:
            message = "Unknown mode value: {}".format(mode)
            raise ValueError(message)

    def __getitem__(self, key):

        value = self.parameters.get(key, OrderedDict())

        return value

    def __setitem__(self, key, value):

        self.parameters.__setitem__(key, value)

        return

    def __contains__(self, key):

        is_contained = key in self.parameters

        return is_contained

    def __iter__(self):

        iterator = iter(self.parameters.keys())

        return iterator

    def __str__(self):

        string = ["{"] + ["{}: {},".format(key, self[key]) for key in self] + ["}"]
        string = "".join(string)

        return string

    def add(self, section, option, value):

        self.parameters[section][option] = value

        return

    def update(self, parameters):

        self.parameters.update(parameters.parameters)

        return

    def save(self, path, **kwargs):

        # Normalize path.
        path = normalize_path(path, **kwargs)

        # Complete path (if necessary).
        if path[-4:] != ".txt":
            path = os.path.join(path, "parameters.txt")

        # Make directory (if necessary).
        directory = os.path.dirname(path)
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # Prepare lines to be saved.
        lines = []
        for section in self.parameters.keys():
            line = "[{}]\n".format(section)
            lines.append(line)
            for option in self.parameters[section].keys():
                if option != 'current_directory' and not self._is_general(section, option):
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

    def _is_general(self, section, option):

        is_general = section != 'general' \
                     and 'general' in self.parameters \
                     and option in self.parameters['general'] \
                     and self.parameters['general'][option] == self.parameters[section][option]

        return is_general
