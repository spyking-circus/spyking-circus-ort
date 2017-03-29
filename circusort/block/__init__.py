import os



def list_modules():
    '''TODO add docstring'''

    path = os.path.abspath(__file__)
    path = os.path.dirname(path)

    modules = []
    for filename in os.listdir(path):
        if filename[-3:] == '.py':
            module = filename[:-3]
            if module[:2] != '__' or module[-2:] != '__':
                modules.append(module)

    return modules


__all__ = list_modules()

from . import *
