from .director import Director




def create_director(interface='127.0.0.1', **kwargs):
    '''Create a new director in this process.'''
    director = Director(interface, **kwargs)
    return director
