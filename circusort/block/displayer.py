# import matplotlib
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mttkinter import mtTkinter as Tk

from circusort.block.tk_block import TkBlock


# matplotlib.use('TkAgg')


class Displayer(TkBlock):
    """Displayer

    Warning: this block is deprecated, matplotlib does not provide a sufficiently fast and scalable visualization.
    """

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
        self._figures = {}
        self._axes = {}
        self._figure_canvasses = {}
        self._backgrounds = {}
        self._lines = {}

        self._number = None
        self._batch = None

    def _initialize(self):

        return

    def _tk_initialize(self):

        self._root = Tk.Tk()

        # Add number label.
        self._label_number = Tk.Label(self._root, text="<number>")
        self._label_number.pack()  # TODO understand meaning.

        # Add batch size label.
        self._label_batch_shape = Tk.Label(self._root, text="<batch shape>")
        self._label_batch_shape.pack()

        # Add figures.
        self._frame = Tk.Frame(self._root)
        k_max = 2 * 4 * 4 * self._nb_channels
        for k in range(0, k_max):
            self._figures[k] = Figure(figsize=(0.5, 0.5), dpi=100)
            self._axes[k] = self._figures[k].add_subplot(1, 1, 1)
            self._axes[k].set_ylim(-50.0, +50.0)
            self._figure_canvasses[k] = FigureCanvasTkAgg(self._figures[k], master=self._frame)
            self._figure_canvasses[k].show()
            self._backgrounds[k] = self._figure_canvasses[k].copy_from_bbox(self._axes[k].bbox)
            x = np.linspace(0.0, 1.0, num=self._nb_samples / 8)
            y = np.zeros_like(x)
            self._lines[k], = self._axes[k].plot(x, y)
            # self._figure_canvasses[k].get_tk_widget().pack()
            nb_columns = 2 * 2 * 2 * 3
            self._figure_canvasses[k].get_tk_widget().grid(row=int(k / nb_columns), column=(k % nb_columns))
        self._frame.pack()

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
        number = data_packet['number']
        batch = data_packet['payload']

        self._measure_time(label='start', period=10)

        # Update number label.
        string = "number: {}"
        text = string.format(number)
        self._label_number.config(text=text)

        # Update batch shape label.
        string = "batch.shape: {}"
        text = string.format(batch.shape)
        self._label_batch_shape.config(text=text)

        # Update figures.
        k_max = 2 * 4 * 4 * self._nb_channels
        for k in range(0, k_max):
            self._lines[k].set_ydata(batch[::8, k % self._nb_channels])
            # # 1st solution:
            # self._figure_canvasses[k].draw()
            # 2nd solution
            self._figures[k].canvas.restore_region(self._backgrounds[k])
            self._axes[k].draw_artist(self._lines[k])
            self._figure_canvasses[k].blit(self._axes[k].bbox)
            self._figure_canvasses[k].flush_events()
            # # 3rd solution
            # self._axes[k].draw_artist(self._axes[k].patch)
            # self._axes[k].draw_artist(self._lines[k])
            # self._figures[k].canvas.update()  # Not supported for FigureCanvasTkAgg.
            # self._figure[k].canvas.flush_events()

        self._root.update()

        self._measure_time(label='end', period=10)

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
