class DictionaryBuffer(object):

    def __init__(self, limit=1000):
        self.data    = []
        self.offsets = [] 
        self.limit   = limit

    def add(self, data):
        if len(self.data) < self.limit:
            offset = data.pop('offset')
            self.data    += [data]
            self.offsets += [offset]
            
    def __len__(self):
        return len(self.data)

    def get(self, idx=0):
        x = self.data[idx]
        y = self.offsets[idx]
        return x, y

    def remove(self, idx=0):
        self.data.pop(idx)
        self.offsets.pop(idx)