import numpy as np

from vispy import app, gloo
from vispy.util import keys

from circusort.io.probe import load_probe
from circusort.io.template import load_template_from_dict
from circusort.obj.cells import Cells
from circusort.obj.cell import Cell

from circusort.block.ort_displayer.utils.widgets import Controler
from circusort.block.ort_displayer.views.canvas import ViewCanvas
from circusort.block.ort_displayer.views.programs import LinesPlot

import sys
import matplotlib.pyplot as plt

TEMPLATE_VERT_SHADER = """
// Index of the signal.
attribute float a_template_index;
// Coordinates of the position of the signal.
attribute vec2 a_template_position;
// Value of the signal.
attribute float a_template_value;
// Color of the signal.
attribute vec3 a_template_color;
// Index of the sample of the signal.
attribute float a_sample_index;
// Bool for template selection.
attribute float a_template_selected;
varying float v_template_selected;
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
// Vertex shader.
void main() {
    // Compute the x coordinate from the sample index.
    float x = +1.0 + 2.0 * u_t_scale * (-1.0 + (a_sample_index / (u_nb_samples_per_signal - 1.0)));
    // Compute the y coordinate from the signal value.
    float y =  a_template_value / u_v_scale;
    // Compute the position.
    vec2 p = vec2(x, y);
    // Affine transformation for the subplots.
    float w = u_x_max - u_x_min;
    float h = u_y_max - u_y_min;
    vec2 a = vec2(1.0 / (1.0 + w / u_d_scale), 1.0 / (1.0 + h / u_d_scale));
    vec2 b = vec2(-1.0 + 2.0 * (a_template_position.x - u_x_min) / w, -1.0 + 2.0 * (a_template_position.y - u_y_min) / h);
    vec2 p_ = a * p + b;
    // Compute GL position.
    gl_Position = vec4(p_, 0.0, 1.0);
    // TODO remove the following;
    v_index = a_template_index;
    v_color = vec4(a_template_color, 1.0);
    v_position = p;
    v_template_selected = a_template_selected;
}
"""

TEMPLATE_FRAG_SHADER = """
// Varying variables.
varying vec4 v_color;
varying float v_index;
varying vec2 v_position;
varying float v_template_selected;
// Fragment shader.
void main() {
    gl_FragColor = v_color;
    // Discard non selected templates;
    if (v_template_selected == 0.0)
        discard;
    // Discard the fragments between the signals (emulate glMultiDrawArrays).
    if (fract(v_index) > 0.0)
        discard;
    // Clipping test.
    if ((abs(v_position.x) > 1.0) || (abs(v_position.y) > 1))
        discard;
}
"""


class TemplateCanvas(ViewCanvas):

    requires = ['templates']
    name = "Templates"

    def __init__(self, probe_path=None, params=None):

        ViewCanvas.__init__(self, probe_path, title="Template View", box='multi')

        nb_buffers_per_signal = int(np.ceil((params['time']['max'] * 1e-3) * params['sampling_rate']
                                            / float(params['nb_samples'])))
        self.nb_buffers_per_signal = nb_buffers_per_signal
        self._time_max = (float(nb_buffers_per_signal * params['nb_samples']) / params['sampling_rate']) * 1e+3
        self._time_min = params['time']['min']
        # self.templates = params['templates']
        self.initialized = False

        self.cells = None

        # Reception
        self.nb_templates = 0
        self.nb_samples_per_template = 0
        self.nb_channels = self.probe.nb_channels
        self.template_values = np.zeros((1, 1), dtype=np.float32)
        self.nb_electrode, self.nb_samples_per_template = 0, 0
        self.nb_data_points = self.nb_channels * self.nb_samples_per_template

        self.templates = np.zeros(shape=(self.nb_data_points * self.nb_templates,), dtype=np.float32)

        self.templates_index = np.repeat((np.arange(0, self.nb_templates, dtype=np.float32)),
                                         repeats=self.nb_data_points)

        self.electrode_index = np.tile(np.repeat(np.arange(0, self.nb_channels, dtype=np.float32),
                                                 repeats=self.nb_samples_per_template),
                                       reps=self.nb_templates)

        self.template_sample_index = np.tile(np.arange(0, self.nb_samples_per_template, dtype=np.float32),
                                             reps=self.nb_templates * self.nb_channels)

        self.template_selected = np.ones(self.nb_templates * self.nb_data_points, dtype=np.float32)

        # Signals.

        # Number of signals.
        self.nb_signals = self.probe.nb_channels
        # Number of samples per buffer.
        self._nb_samples_per_buffer = params['nb_samples']
        # Number of samples per signal.
        nb_samples_per_signal = nb_buffers_per_signal * self._nb_samples_per_buffer
        self._nb_samples_per_signal = nb_samples_per_signal
        # Generate the signal values.
        self._template_values = np.zeros((self.nb_signals, nb_samples_per_signal), dtype=np.float32)

        # Color of each vertex.
        # TODO: make it more efficient by using a GLSL-based color map and the index.
        template_colors = 0.75 * np.ones((self.nb_signals, 3), dtype=np.float32)
        template_colors = np.repeat(template_colors, repeats=nb_samples_per_signal, axis=0)
        template_indices = np.repeat(np.arange(0, self.nb_signals, dtype=np.float32), repeats=nb_samples_per_signal)
        template_positions = np.c_[
            np.repeat(self.probe.x.astype(np.float32), repeats=self.nb_samples_per_template),
            np.repeat(self.probe.y.astype(np.float32), repeats=self.nb_samples_per_template),
        ]
        sample_indices = np.tile(np.arange(0, nb_samples_per_signal, dtype=np.float32),
                                 reps=self.nb_signals)

        self.template_position = np.tile(template_positions, (self.nb_templates, 1))
        self.template_colors = np.repeat(self.get_colors(self.nb_templates),
                                         self.nb_data_points
                                         , axis=0).astype(np.float32)
        self.list_selected_templates = []

        # Define GLSL program.
        self.programs['templates'] = LinesPlot(TEMPLATE_VERT_SHADER, TEMPLATE_FRAG_SHADER)
        self.programs['templates']['a_template_index'] = self.electrode_index
        self.programs['templates']['a_template_position'] = self.template_position
        self.programs['templates']['a_template_value'] = self.templates
        self.programs['templates']['a_template_color'] = self.template_colors
        self.programs['templates']['a_sample_index'] = self.template_sample_index
        self.programs['templates']['a_template_selected'] = self.template_selected
        self.programs['templates']['u_nb_samples_per_signal'] = self.nb_samples_per_template
        self.programs['templates']['u_x_min'] = self.probe.x_limits[0]
        self.programs['templates']['u_x_max'] = self.probe.x_limits[1]
        self.programs['templates']['u_y_min'] = self.probe.y_limits[0]
        self.programs['templates']['u_y_max'] = self.probe.y_limits[1]
        self.programs['templates']['u_d_scale'] = self.probe.minimum_interelectrode_distance
        self.programs['templates']['u_t_scale'] = self._time_max / params['time']['init']
        self.programs['templates']['u_v_scale'] = params['voltage']['init']

        # Final details.
        self.controler = TemplateControler(self, params)


    def on_mouse_wheel(self, event):

        modifiers = event.modifiers

        if keys.CONTROL in modifiers:
            dx = np.sign(event.delta[1]) * 0.01
            v_scale = self.programs['templates']['u_v_scale']
            v_scale_new = v_scale * np.exp(dx)
            self.programs['templates']['u_v_scale'] = v_scale_new
        elif keys.SHIFT in modifiers:
            time_ref = self._time_max
            dx = np.sign(event.delta[1]) * 0.01
            t_scale = self.programs['templates']['u_t_scale']
            t_scale_new = t_scale * np.exp(dx)
            t_scale_new = max(t_scale_new, time_ref / self._time_max)
            t_scale_new = min(t_scale_new, time_ref / self._time_min)
            self.programs['templates']['u_t_scale'] = t_scale_new
        else:
            dx = np.sign(event.delta[1]) * 0.01
            x_min_new = self.programs['templates']['u_x_min'] * np.exp(dx)
            x_max_new = self.programs['templates']['u_x_max'] * np.exp(dx)
            self.programs['templates']['u_x_min'] = x_min_new
            self.programs['templates']['u_x_max'] = x_max_new
            self.programs['box']['u_x_min'] = x_min_new
            self.programs['box']['u_x_max'] = x_max_new

            y_min_new = self.programs['templates']['u_y_min'] * np.exp(dx)
            y_max_new = self.programs['templates']['u_y_max'] * np.exp(dx)
            self.programs['templates']['u_y_min'] = y_min_new
            self.programs['templates']['u_y_max'] = y_max_new
            self.programs['box']['u_y_min'] = y_min_new
            self.programs['box']['u_y_max'] = y_max_new

        # # TODO emit signal to update the spin box.

        self.update()

        return

    def on_mouse_move(self, event):

        if event.press_event is None:
            return

        modifiers = event.modifiers
        p1 = event.press_event.pos
        p2 = event.pos

        p1 = np.array(event.last_event.pos)[:2]
        p2 = np.array(event.pos)[:2]

        dx, dy = 0.1 * (p1 - p2)

        self.programs['box']['u_x_min'] += dx
        self.programs['box']['u_x_max'] += dx
        self.programs['box']['u_y_min'] += dy
        self.programs['box']['u_y_max'] += dy

        self.programs['templates']['u_x_min'] += dx
        self.programs['templates']['u_x_max'] += dx
        self.programs['templates']['u_y_min'] += dy
        self.programs['templates']['u_y_max'] += dy

        # # TODO emit signal to update the spin box.

        self.update()

        return

    # TODO : Warning always called
    def _on_reception(self, data):

        templates = data['templates'] if 'templates' in data else None
        
        if templates is not None:

            for i in range(len(templates)):
                template = load_template_from_dict(templates[i], self.probe)
                data = template.first_component.to_dense()
                if not self.initialized:
                    self.nb_templates = 1
                    self.nb_channels, self.nb_samples_per_template = data.shape[0], data.shape[1]
                    self.template_values = data.ravel().astype(np.float32)
                    self.initialized = True
                    self.nb_data_points = self.nb_channels * self.nb_samples_per_template
                    self.template_positions = np.c_[
                        np.repeat(self.probe.x.astype(np.float32), repeats=self.nb_samples_per_template),
                        np.repeat(self.probe.y.astype(np.float32), repeats=self.nb_samples_per_template),
                    ]
                    self.list_selected_templates = [1] * self.nb_templates
                else:
                    new_template = data.ravel().astype(np.float32)
                    self.template_values = np.concatenate((self.template_values, new_template))
                    self.list_selected_templates.append(0)
                    self.nb_templates += 1

            self.electrode_index = np.tile(np.repeat(np.arange(0, self.nb_channels, dtype=np.float32),
                                                     repeats=self.nb_samples_per_template),
                                           reps=self.nb_templates)
            self.template_position = np.tile(self.template_positions, (self.nb_templates, 1))

            self.template_sample_index = np.tile(np.arange(0, self.nb_samples_per_template, dtype=np.float32),
                                                 reps=self.nb_templates * self.nb_channels)

            self.template_colors = np.repeat(self.get_colors(self.nb_templates),
                                             self.nb_data_points, axis=0).astype(np.float32)

            self.template_selected = np.repeat(self.list_selected_templates,
                                               repeats=self.nb_data_points).astype(np.float32)

            self.programs['templates']['a_template_index'] = self.electrode_index
            self.programs['templates']['a_template_position'] = self.template_position
            self.programs['templates']['a_template_value'] = self.template_values
            self.programs['templates']['a_template_color'] = self.template_colors
            self.programs['templates']['a_sample_index'] = self.template_sample_index
            self.programs['templates']['a_template_selected'] = self.template_selected
            self.programs['templates']['u_nb_samples_per_signal'] = self.nb_samples_per_template

        return

    def _set_value(self, key, value):

        if key == "time":
            t_scale = self._time_max / value
            self.programs['templates']['u_t_scale'] = t_scale
        elif key == "voltage":
            self.programs['templates']['u_v_scale'] = value
        elif key == "templates":
            self.templates = value


    def _highlight_selection(self, selection):
        self.list_selected_templates = [0] * self.nb_templates
        for i in selection:
            self.list_selected_templates[i] = 1
        self.template_selected = np.repeat(self.list_selected_templates,
                                           repeats=self.nb_data_points).astype(np.float32)
        self.programs['templates']['a_template_selected'] = self.template_selected

        return


class TemplateControler(Controler):
    
    def __init__(self, canvas, params):

        Controler.__init__(self, canvas)
        self.params = params
        self.dsb_time = self.double_spin_box(label='time', unit='ms', min_value=params['time']['min'],
                                             max_value=params['time']['max'])

        self.dsb_voltage = self.double_spin_box(label='voltage', unit='µV', min_value=params['voltage']['min'],
                                                max_value=params['voltage']['max'],
                                                init_value=params['voltage']['init'])
        # Signals
        
        self.add_widget(self.dsb_time, self._on_time_changed)
        self.add_widget(self.dsb_voltage, self._on_voltage_changed)

    def _on_time_changed(self):
        time = self.dsb_time['widget'].value()
        self.canvas.set_value({"time" : time})
        return

    def _on_voltage_changed(self):
        voltage = self.dsb_voltage['widget'].value()
        self.canvas.set_value({"voltage" : voltage})
        return
