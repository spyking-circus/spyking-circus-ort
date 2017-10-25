import os



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


def get_config_dirname():
    '''
    Get configuration directory name.

    Return
    ------
    path: string
        Path to the configuration directory '~/.spyking-circus-ort'.
    '''

    path = os.path.join("~", ".spyking-circus-ort")
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)

    return path

def get_tmp_dirname():
    '''
    Get temporary directory name.

    Return
    ------
    path: string
        Path to the temporary directory '~/.spyking-circus-ort/tmp'.
    '''

    path = get_config_dirname()
    path = os.path.join(path, "tmp")
    if not os.path.exists(path):
        os.makedirs(path)

    return path
