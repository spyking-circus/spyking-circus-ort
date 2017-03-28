class Proxy(object):

    address = None
    obj_id = None
    obj_type = None
    attributes = None
    process = None

    def __init__(self, address, obj_id, obj_type, attributes=(), process=None, **kwds):

        object.__init__(self)

        self.address = address
        self.obj_id = obj_id
        self.obj_type = obj_type

        self.attributes = attributes
        self.process = process

    def __repr__(self):

        proxy_repr = '.'.join((self.attributes))
        formatter = '<Proxy for {a}[{i}] {r} >'

        return formatter.format(a=self.address, i=self.obj_id, r=proxy_repr)

    # def __repr__(self):
    #
    #     proxy = self.__getattr__("__repr__")
    #
    #     return proxy()

    def __dict__(self):

        keys = [
            'address',
            'obj_id',
            'obj_type',
            'attributes',
            'process',
        ]

        return keys

    def __getattr__(self, name):

        if name in ['__members__', '__methods__']:
            result = object.__getattr__(self, name)
            # TODO check if this is a proper solution to handle these deprecated attributes
        else:
            result = self.process.get_attr(self, name)

        return result

    def __setattr__(self, name, value):

        if name in dir(self):
            result = object.__setattr__(self, name, value)
        else:
            result = self.process.set_attr(self, name, value)

        return result

    def __call__(self, *args, **kwds):

        result = self.process.call_obj(self, args, kwds)

        return result

    def encode(self):
        '''Serialize proxy.

        TODO complete
        '''

        obj = {
            '__type__': 'proxy',
            'address': self.address,
            'obj_id': self.obj_id,
            'obj_type': self.obj_type,
            'attributes': self.attributes
        }

        return obj
