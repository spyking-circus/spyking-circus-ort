import numpy as np

from circusort.obj.snippet import Snippet


class Snippets(object):

    def __init__(self):

        self._snippets = []

    def __len__(self):

        return len(self._snippets)

    def add(self, element):

        assert isinstance(element, Snippet), "type(element): {}".format(type(element))
        snippet = element
        self._snippets.append(snippet)

        return

    def to_array(self, indices=None):

        list_ = [
            snippet.to_array(indices)
            for snippet in self._snippets
        ]
        array = np.stack(list_)

        return array

    def filter(self):
        hanning_filter = np.hanning(self._snippets[0]._width)
        for snippet in self._snippets:
            snippet.filter(hanning_filter)
