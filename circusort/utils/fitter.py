import numpy as np

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    indices = np.argpartition(ary, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return indices