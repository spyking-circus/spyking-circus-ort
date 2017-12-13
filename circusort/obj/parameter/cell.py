import os

from collections import OrderedDict as odict

from .base import Parameters


class CellParameters(Parameters):
    # TODO add docstring.

    def __init__(self):
        # TODO add docstring.

        Parameters.__init__(self)

        self.parameters = odict([
            ('template', odict()),
            ('train', odict()),
            ('position', odict()),
        ])

    def add(self, section, option, value):
        # TODO add docstring.

        self.parameters[section][option] = value

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
