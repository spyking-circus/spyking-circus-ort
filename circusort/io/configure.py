import os
import ConfigParser as configparser
import re



# CONFIGURATION_PATH = "~/.config/spyking-circus-ort/base.conf"
# CONFIGURATION_PATH = "~/.config/spyking-circus-ort/hosts.conf"
CONFIGURATION_PATH = "~/.config/spyking-circus-ort"
CONFIGURATION_FILES = {
    'base': "base.conf",
    'hosts': "hosts.conf",
}

def load_configuration():
    '''TODO add docstring...'''
    path = CONFIGURATION_PATH
    path = os.path.expanduser(path)
    if os.path.exists(path):
        config = Configuration(path=path)
    else:
        config = Configuration()
    return config

def create():
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



class Host(object):
    '''TODO add docstring...'''
    def __init__(self, host):
        if '@' in host:
            host = host.split('@')
            self.name = host[1]
            self.username = host[0]
        else:
            self.name = host
            self.username = None

    def __repr__(self):
        fmt = "Host (name: {}, username: {})"
        return fmt.format(self.name, self.username)

class HostsParser(object):
    '''TODO add doctring...'''
    def __init__(self):
        self.parameters = dict()

    def read(self, path):
        self.path = path
        f = open(self.path, mode='r')
        text = f.read()
        pattern = re.compile('\S+') # \S matches any non-whitspace character
        hosts = pattern.findall(text)
        f.close()
        hosts = [Host(host) for host in hosts]
        self.parameters['hosts'] = hosts
        return

class Configuration(object):
    '''TODO add docstring...'''
    __default_settings__ = {
        'acquisition': {
            'server_interface': "*"
        },
        'deamon': {
            'protocol': None,
            'interface': None,
            'port': None
        }
    }

    def __init__(self, path=None):
        for section_key, section_value in self.__default_settings__.items():
            section_value = ConfigurationSection(section_value)
            setattr(self, section_key, section_value)
        if path is not None:
            self.path = path
            # Parse base configuration
            self.parser = configparser.ConfigParser()
            base_path = os.path.join(self.path, CONFIGURATION_FILES['base'])
            self.parser.read(base_path)
            for section_key in self.parser.sections():
                section_value = self.parser.items(section_key)
                section_value = dict(section_value)
                section_value = ConfigurationSection(section_value)
            setattr(self, section_key, section_value)
            # Parser hosts configuration
            hosts_parser = HostsParser()
            hosts_path = os.path.join(self.path, CONFIGURATION_FILES['hosts'])
            hosts_parser.read(hosts_path)
            self.update(hosts_parser.parameters)

    def __repr__(self):
        l = ["[{}]\n{}".format(key, getattr(self, key)) for key in self.__default_settings__.keys()]
        s = '\n\n'.join(l)
        return s

    def list_sections(self):
        sections_list = self.__default_settings__.keys()
        return sections_list

    def list_options(self):
        sections_list = self.list_sections()
        options_list = [(section_key, getattr(self, section_key).list_options()) for section_key in sections_list]
        options_list = dict(options_list)
        return options_list

    def update(self, parameters):
        for key, value in parameters.iteritems():
            if isinstance(value, dict):
                value = ConfigurationSection(value)
            setattr(self, key, value)
        return


class ConfigurationSection(object):
    '''TODO add docstring...'''
    def __init__(self, section):
        self.section = section
        for option_key, option_value in section.items():
            setattr(self, option_key, option_value)

    def __repr__(self):
        l = ["{} = {}".format(key, value) for key, value in self.section.items()]
        s = '\n'.join(l)
        return s

    def list_options(self):
        options_list = self.section.keys()
        return options_list
