import json
import logging
import re
import subprocess
import time
import zmq


class LogHandler(logging.Handler):
    """Logging handler sending logging output to a ZMQ socket."""

    def __init__(self, address):

        logging.Handler.__init__(self)

        self.address = address

        self.form = "%(levelname)s %(process)s %(name)s: %(message)s"
        self.formatter = logging.Formatter(self.form)

        # Set up logging connection
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(self.address)

        # TODO: make sure we are connected...
        #       (see http://stackoverflow.com/questions/23433037/sleep-after-zmq-connect#23437374)
        duration = 0.1  # s
        time.sleep(duration)

    def __del__(self):

        self.socket.close()
        self.context.term()
        super(LogHandler, self).close()

    def format(self, record):
        """Format a record."""

        return self.formatter.format(record)

    def emit(self, record):
        """Emit a log message on the PUB socket."""

        topic = b'log'
        message = {
            'kind': 'log',
            'record': record.__dict__,
        }
        data = json.dumps(message)
        data = data.encode('utf-8')  # convert string to bytes
        self.socket.send_multipart([topic, data])

        return

    def handle(self, record):

        super(LogHandler, self).handle(record)

        return


def get_log(address, name=None, log_level=logging.INFO):
    """Get a logger instance by name."""

    # Initialize the handler instance
    handler = LogHandler(address)

    # Get a logger with the specified name (or the root logger)
    logger = logging.getLogger(name=name)
    # Set the threshold for this logger
    logger.setLevel(log_level)
    # logger.setLevel(logging.INFO)
    # Add the specified handler to this logger
    logger.addHandler(handler)

    return logger


def find_interface_address_towards(host):

    p = subprocess.Popen(["ip", "route", "get", host], stdout=subprocess.PIPE)
    lines = p.stdout.readlines()
    s = lines[0]
    k = 'address'
    p = "src (?P<{}>[\d,.]*)".format(k)
    m = re.compile(p)
    try:
        # Python 2 compatibility.
        r = m.search(s)
    except TypeError:
        # Python 3 compatibility.
        s = s.decode('utf-8')
        r = m.search(s)
    a = r.group(k)

    return a
