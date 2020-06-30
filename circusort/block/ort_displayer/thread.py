try:
    from PyQt4.QtCore import QThread, pyqtSignal  # Python 2 compatibility.
except ImportError:  # i.e. ModuleNotFoundError
    from PyQt5.QtCore import QThread, pyqtSignal  # Python 3 compatibility.

class ThreadORT(QThread):

    reception_signal = pyqtSignal(object)

    def __init__(self, all_queues, sleep_duration=None):

        QThread.__init__(self)

        self.queues = {}
        self.sleep_duration = sleep_duration

        for key, value in all_queues.items():
            self.queues[key] = value

    def __del__(self):

        self.wait()

    def run(self):

        while True:

            to_send = {}
            for key, value in self.queues.items():
                if key != 'params':
                    to_send[key] = value.get()

            # Emit signal.
            self.reception_signal.emit(to_send)
            # Sleep.
            self.msleep(self.sleep_duration)  # TODO compute this duration (sampling rate & number of samples per buffer)

        return
