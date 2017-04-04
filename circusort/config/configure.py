import os
import logging
import sys
import ConfigParser as configparser
from circusort.config.probe import Probe

logger = logging.getLogger(__name__)


CONFIGURATION_PATH = "~/.config/spyking-circus-ort/base.conf"

def load_configuration():
    '''TODO add docstring...'''
    path = CONFIGURATION_PATH
    path = os.path.expanduser(path)
    if os.path.exists(path):
        config = Configuration(path=path)
    else:
        config = Configuration()
    return config

def create_configuration():
    '''TODO add docstring...'''
    path = CONFIGURATION_PATH
    path = os.path.expanduser(path)
    if os.path.exists(path):
        raise Exception("File '{}' already exists.".format(path))
    else:
        # Create directories if necessary
        directory_path = os.path.dirname(path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        # Create file
        f = open(path, 'w')
        f.close()
    return

def add(section, option, value):
    '''TODO add docstring...'''
    path = CONFIGURATION_PATH
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        create()
    parser = configparser.ConfigParser()
    parser.read(path)
    if not parser.has_section(section):
        parser.add_section(section)
    parser.set(section, option, value)
    f = open(path, 'wb')
    parser.write(f)
    f.close()
    return

def remove_option(section, option):
    '''TODO add docstring...'''
    path = CONFIGURATION_PATH
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise Exception("File '{}' does not exist.".format(path))
    parser = configparser.ConfigParser()
    parser.read(path)
    if not parser.has_section(section):
        raise Exception("Section '{}' does not exist.".format(section))
    if not parser.has_option(section, option):
        raise Exception("Option '{}' in section '{}' does not exist.".format(option, section))
    parser.remove_option(section, option)
    f = open(path, 'wb')
    parser.write(f)
    f.close()
    return

def remove_section(section):
    '''TODO add docstring...'''
    path = CONFIGURATION_PATH
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise Exception("File '{}' does not exist.".format(path))
    parser = configparser.ConfigParser()
    parser.read(path)
    if not parser.has_section(section):
        raise Exception("Section '{}' does not exist.".format(section))
    parser.remove_section(section)
    f = open(path, 'wb')
    parser.write(f)
    f.close()
    return

def delete():
    '''TODO add docstring...'''
    path = CONFIGURATION_PATH
    path = os.path.expanduser(path)
    if os.path.exists(path):
        os.remove(path)
    else:
        raise Exception("File '{}' does not exist.".format(path))
    return


class Configuration(object):
    '''TODO add docstring...'''
    __default_settings__ = {
        'acquisition': {
            'server_ip' : "127.0.1.1",
            'port'      : '5557',
            'protocol'  : 'tcp', #Could be ipc if on the same machine
            'buffer'    : '1024',
            'file'      : 'tmp.dat',
            'data_dtype': 'float32'
        }
    }

    __special_objects__ = {
        'data' : {
            'mapping' : Probe
        }
    }

    def __init__(self, filename=None):
        for section_key, section_value in self.__default_settings__.items():
            section_value = ConfigurationSection(section_value)
            setattr(self, section_key, section_value)
        if filename is not None:
            self.path = os.path.abspath(os.path.expanduser(filename))
            if not os.path.exists(self.path):
                logger.error("%s does not exist" %self.path)
            self.parser = configparser.ConfigParser()
            self.parser.read(self.path)
            for section_key in self.parser.sections():
                section_value = self.parser.items(section_key)
                section_value = dict(section_value)
                section_value = ConfigurationSection(section_value)
                self.__default_settings__[section_key] = {}

                if section_key in self.__special_objects__.keys():
                    for key, value in self.__special_objects__[section_key].items():
                        if key in section_value.options:
                            setattr(section_value, key, self.__special_objects__[section_key][key](getattr(section_value, key)))

                setattr(self, section_key, section_value)

    @property
    def sections(self):
        return self.__default_settings__.keys()

    @property
    def options(self):
        options_list  = dict([(section_key, getattr(self, section_key).options) for section_key in self.sections])
        return options_list

    @property
    def values(self):
        values_list  = dict([(section_key, getattr(self, section_key).values) for section_key in self.sections])
        return values_list

    @property
    def nb_channels(self):
        N_e = 0
        for key in self.data.mapping.channel_groups.keys():
            N_e += len(self.data.mapping.channel_groups[key]['channels'])

        return N_e

class ConfigurationSection(object):
    '''TODO add docstring...'''
    def __init__(self, section):
        self._section = section
        for option_key, option_value in section.items():
            value    = option_value.split('#')[0].replace(' ', '').replace('\t', '')
            if value.lower() in ['true', 'false']:
                value = bool(value)
            else:
                try:
                    value = float(value)
                except Exception:
                    pass
            setattr(self, option_key, value)

    @property
    def options(self):
        options_list = self._section.keys()
        return options_list

    @property
    def values(self):
        values_list = dict([(option_key, getattr(self, option_key)) for option_key in self.options])
        return values_list
