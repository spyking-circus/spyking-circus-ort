import zmq
import logging
import os
import logging
import thread

logger = logging.getLogger(__name__)

class Node(object):

    def __init__(self, config, name=None):
        self.config = config
        if name is None:
            self.name = "CircusNode"

    def run(self):
        thread.start_new_thread(self._start, ())

    def _start(self):
        raise NotImplementedError