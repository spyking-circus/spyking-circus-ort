import math
import numpy as np

from vispy import app, gloo

from circusort.io.probe import load_probe


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
// ...
attribute float a_signal_index;
// ...
attribute vec2 a_signal_position;
// ...
attribute float a_signal_value;
// ...
attribute vec3 a_signal_color;
// ...
attribute float a_sample_index;
// Number of samples per signal.
uniform float u_nb_samples_per_signal;
// ...
uniform float u_x_min;
uniform float u_x_max;
uniform float u_y_min;
uniform float u_y_max;
uniform float u_t_scale;
uniform float u_v_scale;
// Varying variables used for clipping in the fragment shader.
varying float v_index;
varying vec4 v_color;
varying vec2 v_position;
varying vec4 v_ab;

void main() {
    // Compute the x coordinate from the sample index.
    float x = 10.0 * (-1.0 + 2.0 * a_sample_index / (u_nb_samples_per_signal - 1.0));
    // Compute the y coordinate from the signal value.
    float y =  10.0 * (a_signal_value / 100.0);
    // Compute the position.
    vec2 p = vec2(x, y);
    // Affine transformation for the subplots.
    float w = 0.5 * (u_x_max - u_x_min);
    float h = 0.5 * (u_y_max - u_y_min);
    vec2 a = vec2(1.0 / w, 1.0 / h);
    vec2 b = vec2(a_signal_position.x / w, a_signal_position.y / h);
    vec2 p_ = a * p + b;
    // Compute GL position.
    gl_Position = vec4(p_, 0.0, 1.0);
    // Affine transformation for the entire plot?
    // ...
    // TODO remove the following;
    v_index = a_signal_index;
    v_color = vec4(a_signal_color, 1.0);
    v_position = gl_Position.xy;
    v_ab = vec4(a, b);
}
"""

# VERT_SHADER = """
# // y coordinate of the position.
# attribute float a_position;
# // row, col, and time index.
# attribute vec3 a_index;
# varying vec3 v_index;
# // 2D scaling factor (zooming).
# uniform vec2 u_scale;
# // Size of the table.
# uniform vec2 u_size;
# // Number of samples per signal.
# uniform float u_n;
# // Color.
# attribute vec3 a_color;
# varying vec4 v_color;
# // Varying variables used for clipping in the fragment shader.
# varying vec2 v_position;
# varying vec4 v_ab;
#
# void main() {
#     float nrows = u_size.x;
#     float ncols = u_size.y;
#     // Compute the x coordinate from the time index.
#     float x = -1 + 2*a_index.z / (u_n-1);
#     vec2 position = vec2(x - (1 - 1 / u_scale.x), a_position);
#     // Find the affine transformation for the subplots.
#     vec2 a = vec2(1.0 / ncols, 1.0 / nrows) * 0.9;
#     vec2 b = vec2(-1.0 + 2.0 * (a_index.x + 0.5) / ncols, -1.0 + 2.0 * (a_index.y + 0.5) / nrows);
#     // Apply the static subplot transformation + scaling.
#     gl_Position = vec4(a * u_scale * position + b, 0.0, 1.0);
#     v_color = vec4(a_color, 1.0);
#     v_index = a_index;
#     // For clipping test in the fragment shader.
#     v_position = gl_Position.xy;
#     v_ab = vec4(a, b);
# }
# """

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
varying float v_index;
varying vec2 v_position;
varying vec4 v_ab;
void main() {
    gl_FragColor = v_color;
    // Discard the fragments between the signals (emulate glMultiDrawArrays).
    if (fract(v_index) > 0.0)
        discard;
    // Clipping test.
    //vec2 test = abs((v_position.xy-v_ab.zw)/v_ab.xy);
    //if ((test.x > 1) || (test.y > 1))
    //    discard;
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

    def __init__(self, probe_path=None):

        app.Canvas.__init__(self, title="Vispy canvas", keys="interactive")

        probe = load_probe(probe_path)
        # TODO load all the necessary features from the probe.

        # Number of signals.
        nb_signals = probe.nb_channels
        # Number of samples per signal.
        nb_samples_per_signal = 5 * 1024
        # Generate the signal values.
        self._signal_values = np.zeros((nb_signals, nb_samples_per_signal), dtype=np.float32)
        # Color of each vertex
        # TODO: make it more efficient by using a GLSL-based color map and the index.
        signal_colors = np.repeat(0.75 * np.ones((nb_signals, 3), dtype=np.float32), repeats=nb_samples_per_signal, axis=0)

        signal_indices = np.repeat(np.arange(0, nb_signals, dtype=np.float32), repeats=nb_samples_per_signal)
        signal_positions = np.c_[
            np.repeat(probe.x.astype(np.float32), repeats=nb_samples_per_signal),
            np.repeat(probe.y.astype(np.float32), repeats=nb_samples_per_signal),
        ]
        sample_indices = np.tile(np.arange(0, nb_samples_per_signal, dtype=np.float32), reps=nb_signals)

        self.program = gloo.Program(vert=VERT_SHADER, frag=FRAG_SHADER)
        self.program['a_signal_index'] = signal_indices
        self.program['a_signal_position'] = signal_positions
        self.program['a_signal_value'] = self._signal_values.reshape(-1, 1)
        self.program['a_signal_color'] = signal_colors
        self.program['a_sample_index'] = sample_indices
        self.program['u_nb_samples_per_signal'] = nb_samples_per_signal
        self.program['u_x_min'] = probe.x_limits[0]
        self.program['u_x_max'] = probe.x_limits[1]
        self.program['u_y_min'] = probe.y_limits[0]
        self.program['u_y_max'] = probe.y_limits[1]
        self.program['u_t_scale'] = 1.0
        self.program['u_v_scale'] = 1.0
        # TODO clean/remove the following lines.
        # self.program['a_position'] = y.reshape(-1, 1)
        # self.program['a_color'] = color
        # self.program['a_index'] = index
        # self.program['u_scale'] = (1.0, 0.01)
        # self.program['u_size'] = (nrows, ncols)
        # self.program['u_n'] = n

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

    def on_draw(self, event):

        _ = event
        gloo.clear()
        self.program.draw('line_strip')
        self.program_bis.draw('line_strip')

        return

    def update_data(self, data):

        k = 1024
        self._signal_values[:, :-k] = self._signal_values[:, k:]
        self._signal_values[:, -k:] = np.transpose(data)
        signal_values = self._signal_values.ravel().astype(np.float32)

        self.program['a_signal_value'].set_data(signal_values)
        self.update()

        return
