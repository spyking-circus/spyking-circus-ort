from PyQt4.QtCore import QThread, pyqtSignal


class Thread(QThread):

    number_signal = pyqtSignal(object)
    reception_signal = pyqtSignal(object, object, object)

    def __init__(self, number_pipe, data_pipe, mads_pipe, peaks_pipe):

        QThread.__init__(self)

        self._number_pipe = number_pipe
        self._data_pipe = data_pipe
        self._mads_pipe = mads_pipe
        self._peaks_pipe = peaks_pipe

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
                # Process mads.
                mads = self._mads_pipe[0].recv()
                # Process peaks.
                peaks = self._peaks_pipe[0].recv()
                # Emit signal.
                self.reception_signal.emit(data, mads, peaks)
                # Sleep.
                self.msleep(90)  # TODO compute this duration (sampling rate & number of samples per buffer).
            except EOFError:
                break

        return
