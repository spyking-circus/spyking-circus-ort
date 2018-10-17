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

    def to_array(self):

        list_ = [
            snippet.to_array()
            for snippet in self._snippets
        ]
        array = np.stack(list_)

        return array
