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
