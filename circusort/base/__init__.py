from .director import Director
from .manager import Manager
from .utils import find_interface_address_towards


def create_director(host='127.0.0.1', **kwargs):
    """Create a new director in this process.

    Parameter:
        host: string (optional)
            The IP address of the host of the director.
    Return:
        director: circusort.base.director
            The director.
    See also:
        circusort.base.director.Director
    """

    interface = find_interface_address_towards(host)
    director = Director(interface, **kwargs)

    return director
