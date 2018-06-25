from PyQt4.QtCore import QThread, pyqtSignal


class QtThread(QThread):

    number_signal = pyqtSignal(object)
    data_signal = pyqtSignal(object)

    def __init__(self, number_pipe, data_pipe):

        QThread.__init__(self)

        self._number_pipe = number_pipe
        self._data_pipe = data_pipe

    def __del__(self):

        self.wait()

    def run(self):

        while True:

            try:
                # Process number.
                number = self._number_pipe[0].recv()
                self.number_signal.emit(str(number))
                # Process data.
                data = self._data_pipe[0].recv()
                self.data_signal.emit(data)
                # Sleep.
                self.usleep(100)
            except EOFError:
                break

        return
