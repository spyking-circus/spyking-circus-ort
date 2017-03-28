from .director import Director



def create_director(**kwargs):
    '''Create a new director in this process.'''
    director = Director(**kwargs)
    return director
