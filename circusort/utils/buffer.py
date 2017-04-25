import numpy

class DictionaryBuffer(object):

    def __init__(self, limit=1000):
        self.data    = []
        self.offsets = numpy.zeros(0, dtype=numpy.uint32)
        self.limit   = limit

    def add(self, data):
        offset        = data.pop('offset')
        self.data    += [data]
        self.offsets  = numpy.concatenate((self.offsets, numpy.array([offset], dtype=numpy.uint32)))
            
    def __len__(self):
        return len(self.data)

    def get(self, offset):
        idx = numpy.where(self.offsets == offset)[0]
        if len(idx) > 0:
            return self.data[idx[0]]
        else:
            return None

    def remove(self, offset):
        indices = numpy.where(self.offsets <= offset)[0]
        for idx in indices:
            self.data.pop(idx)
        self.offsets = numpy.delete(self.offsets, indices)