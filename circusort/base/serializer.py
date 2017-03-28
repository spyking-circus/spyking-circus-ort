import json

from circusort.base.proxy import Proxy



class Serializer(object):
    '''TODO add docstring'''

    def __init__(self):

        object.__init__(self)

        class Encoder(json.JSONEncoder):

            def default(self, obj):
                if obj is None:
                    obj = json.JSONEncoder.default(obj)
                else:
                    if isinstance(obj, Proxy):
                        obj = obj.encode()
                    else:
                        raise TypeError("Type {t} is not serializable.".format(t=type(obj)))
                return obj

        self.Encoder = Encoder

    def dumps(self, message):
        '''TODO add docstring'''

        dumped_request_id = str(message['request_id']).encode()
        dumped_request = str(message['request']).encode()
        dumped_serialization_type = str('json').encode()
        if message['options'] is None:
            dumped_options = b""
        else:
            dumped_options = json.dumps(message['options'], cls=self.Encoder).encode()

        message = [
            dumped_request_id,
            dumped_request,
            dumped_serialization_type,
            dumped_options,
        ]

        return message

    def loads(self, message):
        '''TODO add docstring'''

        raise NotImplementedError()
