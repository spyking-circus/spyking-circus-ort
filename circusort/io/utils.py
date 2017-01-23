import os



def get_header_path(path):
    '''TODO add docstring...'''
    path, ext = os.path.splitext(path)
    path = path + ".header"
    return path
