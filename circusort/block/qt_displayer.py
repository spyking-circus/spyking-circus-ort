import math
import numpy as np
import sys

from multiprocessing import Process, Pipe
from PyQt4.QtCore import QThread, Qt, pyqtSignal
from PyQt4.QtGui import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout
from vispy import app, gloo, scene

from circusort.block.block import Block


__classname__ = "QtDisplayer"


class QtDisplayer(Block):
    """Displayer"""

    name = "Displayer"

    params = {}

    def __init__(self, **kwargs):
        """Initialization"""

        Block.__init__(self, **kwargs)
        self.add_input('data', structure='dict')

        self._dtype = None
        self._nb_samples = None
        self._nb_channels = None
        self._sampling_rate = None

        self._pipe = Pipe()
        self._data_pipe = Pipe()
        self._qt_process = QtProcess(self._pipe, self._data_pipe)

    def _initialize(self):

        self._qt_process.start()

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

        self._measure_time(label='start', frequency=10)

        # TODO remove the 2 following lines.
        self._pipe[1].send(number)
        self._data_pipe[1].send(batch)

        self._measure_time(label='end', frequency=10)

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


class QtProcess(Process):

    def __init__(self, pipe, data_pipe):

        Process.__init__(self)

        self._pipe = pipe
        self._data_pipe = data_pipe

    def run(self):

        app = QApplication(sys.argv)
        screen_resolution = app.desktop().screenGeometry()
        window = QtWindow(self._pipe, self._data_pipe, screen_resolution)
        window.show()
        app.exec_()

        return


# class QtMainWindow(QMainWindow):
#
#     def __init__(self, pipe, data_pipe, screen_resolutions):
#
#         QMainWindow.__init__(self)
#
#         self.setCentralWidget(...)  # TODO complete.
#
#         self.addDockWidget(...)  # TODO complete.


class QtWindow(QWidget):

    def __init__(self, pipe, data_pipe, screen_resolution):

        QWidget.__init__(self)

        screen_width = screen_resolution.width()
        screen_height = screen_resolution.height()

        self._label = QLabel()
        self._label.setText("<number>")
        self._label.setAlignment(Qt.AlignCenter)

        self._canvas = VispyCanvas()

        self._vbox = QVBoxLayout()
        self._vbox.addWidget(self._label)
        self._vbox.addWidget(self._canvas.native)

        self.setLayout(self._vbox)

        self.setWindowTitle("SpyKING Circus ORT - Read 'n' display (Qt)")
        self.resize(screen_width, screen_height)

        self._thread = MyQtThread(pipe, data_pipe)

        self._thread.number_signal.connect(self.number_callback)
        self._thread.data_signal.connect(self.data_callback)

        self._thread.start()

    def number_callback(self, number):

        self._label.setText(number)

        return

    def data_callback(self, data):

        self._canvas.update_data(data)

        return


class MyQtThread(QThread):

    number_signal = pyqtSignal(object)
    data_signal = pyqtSignal(object)

    def __init__(self, pipe, data_pipe):

        QThread.__init__(self)

        self._pipe = pipe
        self._data_pipe = data_pipe

    def __del__(self):

        self.wait()

    def run(self):

        while True:

            try:
                number = self._pipe[0].recv()
                self.number_signal.emit(str(number))
                data = self._data_pipe[0].recv()
                self.data_signal.emit(data)
                self.usleep(1)
            except EOFError:
                break

        return


# TODO clean the following lines.

# Number of cols and rows in the table.
nrows = 3
ncols = 3

# Number of signals.
m = nrows * ncols

# Number of samples per signal.
n = 5 * 1024

# Various signal amplitudes.
amplitudes = .1 + .2 * np.random.rand(m, 1).astype(np.float32)

# Generate the signals as a (m, n) array.
y = amplitudes * np.random.randn(m, n).astype(np.float32)

# Color of each vertex (TODO: make it more efficient by using a GLSL-based
# color map and the index).
color = np.repeat(0.75 * np.ones((m, 3)), n, axis=0).astype(np.float32)

# Signal 2D index of each vertex (row and col) and x-index (sample index
# within each signal).
index = np.c_[
    np.repeat(np.repeat(np.arange(ncols), nrows), n),
    np.repeat(np.tile(np.arange(nrows), ncols), n),
    np.tile(np.arange(n), m),
].astype(np.float32)

# Signal 2D index of each vertex (row and col) and corner index.
index_bis = np.c_[
    np.repeat(np.repeat(np.arange(ncols), nrows), 5),
    np.repeat(np.tile(np.arange(nrows), ncols), 5),
    np.tile(np.array([+1, -1, -1, +1, +1]), m),
    np.tile(np.array([+1, +1, -1, -1, +1]), m),
].astype(np.float32)

VERT_SHADER = """
#version 120
// y coordinate of the position.
attribute float a_position;
// row, col, and time index.
attribute vec3 a_index;
varying vec3 v_index;
// 2D scaling factor (zooming).
uniform vec2 u_scale;
// Size of the table.
uniform vec2 u_size;
// Number of samples per signal.
uniform float u_n;
// Color.
attribute vec3 a_color;
varying vec4 v_color;
// Varying variables used for clipping in the fragment shader.
varying vec2 v_position;
varying vec4 v_ab;
void main() {
    float nrows = u_size.x;
    float ncols = u_size.y;
    // Compute the x coordinate from the time index.
    float x = -1 + 2*a_index.z / (u_n-1);
    vec2 position = vec2(x - (1 - 1 / u_scale.x), a_position);
    // Find the affine transformation for the subplots.
    vec2 a = vec2(1.0 / ncols, 1.0 / nrows) * 0.9;
    vec2 b = vec2(-1.0 + 2.0 * (a_index.x + 0.5) / ncols, -1.0 + 2.0 * (a_index.y + 0.5) / nrows);
    // Apply the static subplot transformation + scaling.
    gl_Position = vec4(a * u_scale * position + b, 0.0, 1.0);
    v_color = vec4(a_color, 1.0);
    v_index = a_index;
    // For clipping test in the fragment shader.
    v_position = gl_Position.xy;
    v_ab = vec4(a, b);
}
"""

VERT_SHADER_BIS = """
// row, col and corner index
attribute vec4 a_index;
varying vec4 v_index;
// 2D scaling factor (zooming).
uniform vec2 u_scale;
// Size of the table.
uniform vec2 u_size;
void main() {
    float nrows = u_size.x;
    float ncols = u_size.y;
    // Compute the x coordinate.
    float x = a_index.z;
    // Compute the y coordinate.
    float y = a_index.w;
    // Compute the position.
    vec2 position = vec2(x, y);
    // Find the affine transformation for the subplots.
    vec2 a = vec2(1.0 / ncols, 1.0 / nrows) * 0.9;
    vec2 b = vec2(-1.0 + 2.0 * (a_index.x + 0.5) / ncols, -1.0 + 2.0 * (a_index.y + 0.5) / nrows);
    // Apply the static subplot transformation + scaling.
    gl_Position = vec4(a * u_scale * position + b, 0.0, 1.0);
    v_index = a_index;
}
"""

FRAG_SHADER = """
#version 120
varying vec4 v_color;
varying vec3 v_index;
varying vec2 v_position;
varying vec4 v_ab;
void main() {
    gl_FragColor = v_color;
    // Discard the fragments between the signals (emulate glMultiDrawArrays).
    if ((fract(v_index.x) > 0.) || (fract(v_index.y) > 0.))
        discard;
    // Clipping test.
    vec2 test = abs((v_position.xy-v_ab.zw)/v_ab.xy);
    if ((test.x > 1) || (test.y > 1))
        discard;
}
"""

FRAG_SHADER_BIS = """
varying vec4 v_index;
void main() {
    gl_FragColor = vec4(0.25, 0.25, 0.25, 1.0);
    // Discard the fragments between the box (emulate glMultiDrawArrays).
    if ((v_index.z == +1.0) && (v_index.w == +1.0))
        discard;
}
"""


class VispyCanvas(app.Canvas):

    def __init__(self):

        app.Canvas.__init__(self, title="Vispy canvas", keys="interactive")

        self.program = gloo.Program(vert=VERT_SHADER, frag=FRAG_SHADER)
        self.program['a_position'] = y.reshape(-1, 1)
        self.program['a_color'] = color
        self.program['a_index'] = index
        self.program['u_scale'] = (1.0, 0.01)
        self.program['u_size'] = (nrows, ncols)
        self.program['u_n'] = n

        self.program_bis = gloo.Program(vert=VERT_SHADER_BIS, frag=FRAG_SHADER_BIS)
        self.program_bis['a_index'] = index_bis
        self.program_bis['u_scale'] = (1.0, 1.0)
        self.program_bis['u_size'] = (nrows, ncols)

        gloo.set_viewport(0, 0, *self.physical_size)

        # self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    def on_resize(self, event):

        gloo.set_viewport(0, 0, *event.physical_size)

        return

    def on_mouse_wheel(self, event):

        dx = np.sign(event.delta[1]) * 0.05
        scale_x, scale_y = self.program['u_scale']
        scale_x_new, scale_y_new = (scale_x * math.exp(2.5 * dx), scale_y)
        self.program['u_scale'] = (max(1.0, scale_x_new), scale_y_new)
        scale_x, scale_y = self.program_bis['u_scale']
        scale_x_new, scale_y_new = (scale_x * math.exp(2.5 * dx), scale_y)
        self.program_bis['u_scale'] = (max(1.0, scale_x_new), scale_y_new)
        self.update()

        return

    def on_timer(self, event):
        """Add some data at the end of each signal (real-time signals)."""
        k = 10
        y[:, :-k] = y[:, k:]
        y[:, -k:] = amplitudes * np.random.randn(m, k)

        self.program['a_position'].set_data(y.ravel().astype(np.float32))
        self.update()

    def on_draw(self, event):

        _ = event
        gloo.clear()
        self.program.draw('line_strip')
        self.program_bis.draw('line_strip')

        return

    def update_data(self, data):

        k = 1024
        y[:, :-k] = y[:, k:]
        y[:, -k:] = np.transpose(data)

        self.program['a_position'].set_data(y.ravel().astype(np.float32))
        self.update()

        return
