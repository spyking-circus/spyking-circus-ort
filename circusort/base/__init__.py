from .director import Director



def create_director(interface='127.0.0.1'):
    '''Create a new director in this process.'''
    director = Director(interface=interface)
    return director
