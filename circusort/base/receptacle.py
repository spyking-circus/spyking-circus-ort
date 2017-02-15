class Receptacle(object):

    def __init__(self):
        self.protocol = None
        self.interface = None

    def configure(self, protocol='tcp', interface='127.0.0.1'):
        self.protocol = protocol
        self.interface = interface
        return

    def connect(self, receptacle):
        return
