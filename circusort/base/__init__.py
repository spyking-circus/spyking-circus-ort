from .director import Director
from .manager import Manager

def create_director(host='127.0.0.1', **kwargs):
    '''Create a new director in this process.'''
    interface = find_interface_address_towards(host)
    director = Director(interface, **kwargs)
    return director
