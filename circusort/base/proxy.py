class Proxy(object):

    def __init__(self, address, obj_id, ref_id, obj_type, attributes=(), process=None, **kwds):

        object.__init__(self)

        self.address = address
        self.obj_id = obj_id
        self.ref_id = ref_id
        self.obj_type = obj_type

        self.obj_type = ''
        self.attributes = attributes
        self.process = process

    def __repr__(self):

        proxy_repr = '.'.join((self.obj_type))
        formatter = '<Proxy for {a}[{i}] {r} >'

        return formatter.format(a=self.address, i=self.obj_id, r=proxy_repr)

    def __getattr__(self, attr):

        proxy = Proxy(self.address, self.obj_id, self.ref_id, self.obj_type,
                      attributes=self.attributes + (attr,),
                      process=self.process)
        proxy.__dict__['parent_proxy'] = self

        return proxy

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
            'ref_id': self.ref_id,
            'obj_type': self.obj_type,
            'attributes': self.attributes
        }

        return obj
