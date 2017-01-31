import os
import ConfigParser as configparser



CONFIGURATION_PATH = "~/.config/spyking-circus-ort/base.conf"
CONFIGURATION_PATH = "~/Programming/github/spyking-circus-ort/circusort/io/base.conf"

def isdata(path):
    '''Check if path corresponds to existing regular data

    Parameter
    ---------
    path: string

    Return
    ------
    flag: boolean

    '''
    flag = os.path.isfile(path)
    return flag


def load_configuration():
    '''TODO add docstring...'''
    path = CONFIGURATION_PATH
    path = os.path.expanduser(path)
    config = Configuration(path)
    return config


class Configuration(object):
    '''TODO add docstring...'''
    def __init__(self, path):
        self.path = path
        if os.path.exists(self.path):
            self.parser = configparser.ConfigParser()
            self.parser.read(self.path)
        else:
            raise Exception("Configuration file {} does not exists.".format(self.path))

    def __getattr__(self, key):
        if self.parser.has_section(key):
            section = ConfigurationSection(self, key)
            return section
        print(key)
        return

class ConfigurationSection(object):
    '''TODO add docstring...'''
    def __init__(self, configuration, section):
        self.configuration = configuration
        self.section = section

    def __getattr__(self, key):
        if self.configuration.parser.has_option(self.section, key):
            option = self.configuration.parser.get(self.section, key)
            return option
        else:
            raise AttributeError("Configuration section '{}' has no option '{}'".format(self.section, key))
