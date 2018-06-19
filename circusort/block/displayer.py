import numpy as np

from mttkinter import mtTkinter as Tk

from circusort.block.tk_block import TkBlock


class Displayer(TkBlock):
    """Displayer"""

    name = "Displayer"

    params = {}

    def __init__(self, **kwargs):
        """Initialization"""

        TkBlock.__init__(self, **kwargs)
        self.add_input('data', structure='dict')

        self._dtype = None
        self._nb_samples = None
        self._nb_channels = None
        self._sampling_rate = None

        self._root = None
        self._label_number = None
        self._label_batch_shape = None

        self._number = None
        self._batch = None

    def _initialize(self):

        return

    def _tk_initialize(self):

        self._root = Tk.Tk()
        self._label_number = Tk.Label(self._root, text="<number>")
        self._label_number.pack()  # TODO understand meaning.
        self._label_batch_shape = Tk.Label(self._root, text="<batch shape>")
        self._label_batch_shape.pack()
        self._root.update()

        return

    def _configure_input_parameters(self, dtype=None, nb_samples=None, nb_channels=None, sampling_rate=None, **kwargs):

        self._dtype = dtype
        self._nb_samples = nb_samples
        self._nb_channels = nb_channels
        self._sampling_rate = sampling_rate

        return

    def _process(self):

        data_packet = self.get_input('data').receive()
        self._number = data_packet['number']
        self._batch = data_packet['payload']

        self._measure_time(label='start', frequency=10)

        # TODO improve the following lines.
        string = "number: {}"
        text = string.format(self._number)
        self._label_number.config(text=text)
        string = "batch.shape: {}"
        text = string.format(self._batch.shape)
        self._label_batch_shape.config(text=text)
        self._root.update()

        self._measure_time(label='end', frequency=10)

        return

    def _tk_finalize(self):

        self._root.quit()

        return

    def _introspect(self):

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self._nb_samples) / self._sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        # Log info message.
        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return
