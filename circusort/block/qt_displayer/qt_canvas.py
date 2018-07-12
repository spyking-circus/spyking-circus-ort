import numpy as np

from vispy import app, gloo

from circusort.io.probe import load_probe


SIGNAL_VERT_SHADER = """
// Index of the signal.
attribute float a_signal_index;
// Coordinates of the position of the signal.
attribute vec2 a_signal_position;
// Value of the signal.
attribute float a_signal_value;
// Color of the signal.
attribute vec3 a_signal_color;
// Index of the sample of the signal.
attribute float a_sample_index;
// Number of samples per signal.
uniform float u_nb_samples_per_signal;
// Uniform variables used to transform the subplots.
uniform float u_x_min;
uniform float u_x_max;
uniform float u_y_min;
uniform float u_y_max;
uniform float u_d_scale;
uniform float u_t_scale;
uniform float u_v_scale;
// Varying variables used for clipping in the fragment shader.
varying float v_index;
varying vec4 v_color;
varying vec2 v_position;
varying vec4 v_ab;
// Vertex shader.
void main() {
    // Compute the x coordinate from the sample index.
    float x = +1.0 + 2.0 * u_t_scale * (-1.0 + (a_sample_index / (u_nb_samples_per_signal - 1.0)));
    // Compute the y coordinate from the signal value.
    float y =  a_signal_value / u_v_scale;
    // Compute the position.
    vec2 p = vec2(x, y);
    // Affine transformation for the subplots.
    float w = u_x_max - u_x_min;
    float h = u_y_max - u_y_min;
    vec2 a = vec2(1.0 / (1.0 + w / u_d_scale), 1.0 / (1.0 + h / u_d_scale));
    vec2 b = vec2(-1.0 + 2.0 * (a_signal_position.x - u_x_min) / w, -1.0 + 2.0 * (a_signal_position.y - u_y_min) / h);
    vec2 p_ = a * p + b;
    // Compute GL position.
    gl_Position = vec4(p_, 0.0, 1.0);
    // TODO remove the following;
    v_index = a_signal_index;
    v_color = vec4(a_signal_color, 1.0);
    v_position = p;
    v_ab = vec4(a, b);
}
"""

BOX_VERT_SHADER = """
// Index of the box.
attribute float a_box_index;
// Coordinates of the position of the box.
attribute vec2 a_box_position;
// Coordinates of the position of the corner.
attribute vec2 a_corner_position;
// Uniform variables used to transform the subplots.
uniform float u_x_min;
uniform float u_x_max;
uniform float u_y_min;
uniform float u_y_max;
uniform float u_d_scale;
// Varying variable used for clipping in the fragment shader.
varying float v_index;
// Vertex shader.
void main() {
    // Compute the x coordinate.
    float x = a_corner_position.x;
    // Compute the y coordinate.
    float y = a_corner_position.y;
    // Compute the position.
    vec2 p = a_corner_position;
    // Find the affine transformation for the subplots.
    float w = u_x_max - u_x_min;
    float h = u_y_max - u_y_min;
    vec2 a = vec2(1.0 / (1.0 + w / u_d_scale), 1.0 / (1.0 + h / u_d_scale));
    vec2 b = vec2(-1.0 + 2.0 * (a_box_position.x - u_x_min) / w, -1.0 + 2.0 * (a_box_position.y - u_y_min) / h);
    // Apply the transformation.
    gl_Position = vec4(a * p + b, 0.0, 1.0);
    v_index = a_box_index;
}
"""

SIGNAL_FRAG_SHADER = """
// Varying variables.
varying vec4 v_color;
varying float v_index;
varying vec2 v_position;
varying vec4 v_ab;
// Fragment shader.
void main() {
    gl_FragColor = v_color;
    // Discard the fragments between the signals (emulate glMultiDrawArrays).
    if (fract(v_index) > 0.0)
        discard;
    // Clipping test.
    if ((abs(v_position.x) > 1.0) || (abs(v_position.y) > 1))
        discard;
}
"""

BOX_FRAG_SHADER = """
// Varying variable.
varying float v_index;
// Fragment shader.
void main() {
    gl_FragColor = vec4(0.25, 0.25, 0.25, 1.0);
    // Discard the fragments between the box (emulate glMultiDrawArrays).
    if (fract(v_index) > 0.0)
        discard;
}
"""


class VispyCanvas(app.Canvas):

    def __init__(self, probe_path=None, params=None):

        app.Canvas.__init__(self, title="Vispy canvas", keys="interactive")

        probe = load_probe(probe_path)
        nb_buffers_per_signal = int(np.ceil((params['time']['max'] * 1e-3) * params['sampling_rate'] / float(params['nb_samples'])))
        self._time_max = (float(nb_buffers_per_signal * params['nb_samples']) / params['sampling_rate']) * 1e+3
        self._time_min = params['time']['min']

        # Number of signals.
        nb_signals = probe.nb_channels
        # Number of samples per buffer.
        self._nb_samples_per_buffer = params['nb_samples']
        # Number of samples per signal.
        nb_samples_per_signal = nb_buffers_per_signal * self._nb_samples_per_buffer
        # Generate the signal values.
        self._signal_values = np.zeros((nb_signals, nb_samples_per_signal), dtype=np.float32)
        # Color of each vertex.
        # TODO: make it more efficient by using a GLSL-based color map and the index.
        signal_colors = 0.75 * np.ones((nb_signals, 3), dtype=np.float32)
        signal_colors = np.repeat(signal_colors, repeats=nb_samples_per_signal, axis=0)
        signal_indices = np.repeat(np.arange(0, nb_signals, dtype=np.float32), repeats=nb_samples_per_signal)
        signal_positions = np.c_[
            np.repeat(probe.x.astype(np.float32), repeats=nb_samples_per_signal),
            np.repeat(probe.y.astype(np.float32), repeats=nb_samples_per_signal),
        ]
        sample_indices = np.tile(np.arange(0, nb_samples_per_signal, dtype=np.float32), reps=nb_signals)

        self.program = gloo.Program(vert=SIGNAL_VERT_SHADER, frag=SIGNAL_FRAG_SHADER)
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
        self.program['u_d_scale'] = probe.minimum_interelectrode_distance
        self.program['u_t_scale'] = self._time_max / params['time']['init']
        self.program['u_v_scale'] = params['voltage']['init']

        box_indices = np.repeat(np.arange(0, nb_signals, dtype=np.float32), repeats=5)
        box_positions = np.c_[
            np.repeat(probe.x.astype(np.float32), repeats=5),
            np.repeat(probe.y.astype(np.float32), repeats=5),
        ]
        corner_positions = np.c_[
            np.tile(np.array([+1.0, -1.0, -1.0, +1.0, +1.0], dtype=np.float32), reps=nb_signals),
            np.tile(np.array([+1.0, +1.0, -1.0, -1.0, +1.0], dtype=np.float32), reps=nb_signals),
        ]

        self.program_bis = gloo.Program(vert=BOX_VERT_SHADER, frag=BOX_FRAG_SHADER)
        self.program_bis['a_box_index'] = box_indices
        self.program_bis['a_box_position'] = box_positions
        self.program_bis['a_corner_position'] = corner_positions
        self.program_bis['u_x_min'] = probe.x_limits[0]
        self.program_bis['u_x_max'] = probe.x_limits[1]
        self.program_bis['u_y_min'] = probe.y_limits[0]
        self.program_bis['u_y_max'] = probe.y_limits[1]
        self.program_bis['u_d_scale'] = probe.minimum_interelectrode_distance

        gloo.set_viewport(0, 0, *self.physical_size)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    @staticmethod
    def on_resize(event):

        gloo.set_viewport(0, 0, *event.physical_size)

        return

    # def on_mouse_wheel(self, event):
    #
    #     time_ref = self._time_max
    #
    #     dx = np.sign(event.delta[1]) * 0.05
    #     t_scale = self.program['u_t_scale']
    #     t_scale_new = t_scale * np.exp(2.5 * dx)
    #     t_scale_new = max(t_scale_new, time_ref / self._time_max)
    #     t_scale_new = min(t_scale_new, time_ref / self._time_min)
    #     self.program['u_t_scale'] = t_scale_new
    #
    #     # TODO emit signal to update the spin box.
    #
    #     self.update()
    #
    #     return

    def on_draw(self, event):

        _ = event
        gloo.clear()
        self.program.draw('line_strip')
        self.program_bis.draw('line_strip')

        return

    def update_data(self, data):

        # TODO find a better solution for the 2 following lines.
        if data.shape[1] > 256:
            data = data[:, 0:256]

        k = self._nb_samples_per_buffer
        self._signal_values[:, :-k] = self._signal_values[:, k:]
        self._signal_values[:, -k:] = np.transpose(data)
        signal_values = self._signal_values.ravel().astype(np.float32)

        self.program['a_signal_value'].set_data(signal_values)
        self.update()

        return

    def set_time(self, value):

        t_scale = self._time_max / value
        self.program['u_t_scale'] = t_scale
        self.update()

        return

    def set_voltage(self, value):

        v_scale = value
        self.program['u_v_scale'] = v_scale
        self.update()

        return
