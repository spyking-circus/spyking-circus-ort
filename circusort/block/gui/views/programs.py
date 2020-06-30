from vispy import app, gloo
from vispy.util import keys
import numpy as np


class LinesPlot(gloo.Program):

    def __init__(self, vert, frag):
        gloo.Program.__init__(self, vert=vert, frag=frag)

    def _draw(self):

        self.draw('line_strip')

class ScatterPlot(gloo.Program):

    def __init__(self, vert, frag):
        gloo.Program.__init__(self, vert=vert, frag=frag)

    def _draw(self):

        self.draw('points')


class BoxPlot(LinesPlot):

    VERT_SHADER = """
    attribute vec2 a_position;
    void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
    """

    FRAG_SHADER = """
    // Fragment shader.
    void main() {
        gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0);
    }
    """
    def __init__(self, box_corner_positions=None):
        LinesPlot.__init__(self, self.VERT_SHADER, self.FRAG_SHADER)
        if box_corner_positions is None:
            box_corner_positions = np.array([[+0.9, +0.9],
                                             [-0.9, +0.9],
                                             [-0.9, -0.9],
                                             [+0.9, -0.9],
                                             [+0.9, +0.9]], dtype=np.float32)
            self.set_corners(box_corner_positions)

    def set_corners(self, box_corner_positions):
        self.__setitem__('a_position', gloo.VertexBuffer(box_corner_positions))


class ProbeBoxPlot(LinesPlot):

    FRAG_SHADER = """
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

    VERT_SHADER = """
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


    def __init__(self, probe, box_corner_positions=None):
        
        LinesPlot.__init__(self, self.VERT_SHADER, self.FRAG_SHADER)
        self.probe = probe
        self.set_corners(box_corner_positions)

    def set_corners(self, box_corner_positions):

        box_indices = np.repeat(np.arange(0, self.probe.nb_channels, dtype=np.float32), repeats=5)
        box_positions = np.c_[
            np.repeat(self.probe.x.astype(np.float32), repeats=5),
            np.repeat(self.probe.y.astype(np.float32), repeats=5),
        ]
        if box_corner_positions is None:
            box_corner_positions = np.c_[
                np.tile(np.array([+1.0, -1.0, -1.0, +1.0, +1.0], dtype=np.float32), reps=self.probe.nb_channels),
                np.tile(np.array([+1.0, +1.0, -1.0, -1.0, +1.0], dtype=np.float32), reps=self.probe.nb_channels),
            ]
        # Define GLSL program.
        self.__setitem__('a_box_index', gloo.VertexBuffer(box_indices))
        self.__setitem__('a_box_position', gloo.VertexBuffer(box_positions))
        self.__setitem__('a_corner_position', gloo.VertexBuffer(box_corner_positions))
        self.__setitem__('u_x_min', self.probe.x_limits[0])
        self.__setitem__('u_x_max', self.probe.x_limits[1])
        self.__setitem__('u_y_min', self.probe.y_limits[0])
        self.__setitem__('u_y_max', self.probe.y_limits[1])
        self.__setitem__('u_d_scale', self.probe.minimum_interelectrode_distance)



class ProbeSingleLinePlot(LinesPlot):

    TEMPLATE_VERT_SHADER = """
    // Index of the signal.
    attribute float a_index;
    // Coordinates of the position of the signal.
    attribute vec2 a_positions;
    // Value of the signal.
    attribute float a_values;
    // Color of the signal.
    attribute vec3 a_colors;
    // Index of the sample of the signal.
    attribute float a_sample_index;
    // Bool for template selection.
    attribute float a_selection;
    varying float v_selection;
    // Number of samples per signal.
    uniform float u_nb_points;
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
        float x = +1.0 + 2.0 * u_t_scale * (-1.0 + (a_sample_index / (u_nb_points - 1.0)));
        // Compute the y coordinate from the signal value.
        float y =  a_values / u_v_scale;
        // Compute the position.
        vec2 p = vec2(x, y);
        // Affine transformation for the subplots.
        float w = u_x_max - u_x_min;
        float h = u_y_max - u_y_min;
        vec2 a = vec2(1.0 / (1.0 + w / u_d_scale), 1.0 / (1.0 + h / u_d_scale));
        vec2 b = vec2(-1.0 + 2.0 * (a_positions.x - u_x_min) / w, -1.0 + 2.0 * (a_positions.y - u_y_min) / h);
        vec2 p_ = a * p + b;
        // Compute GL position.
        gl_Position = vec4(p_, 0.0, 1.0);
        // TODO remove the following;
        v_index = a_index;
        v_color = vec4(a_colors, 1.0);
        v_position = p;
        v_selection = a_selection;
    }
    """

    TEMPLATE_FRAG_SHADER = """
    // Varying variables.
    varying vec4 v_color;
    varying float v_index;
    varying vec2 v_position;
    varying float v_selection;
    // Fragment shader.
    void main() {
        gl_FragColor = v_color;
        // Discard non selected templates;
        if (v_selection == 0.0)
            discard;
        // Discard the fragments between the signals (emulate glMultiDrawArrays).
        if (fract(v_index) > 0.0)
            discard;
        // Clipping test.
        if ((abs(v_position.x) > 1.0) || (abs(v_position.y) > 1))
            discard;
    }
    """

    def __init__(self, probe, data=np.zeros((0, 0), dtype=np.float32)):
        LinesPlot.__init__(self, self.VERT_SHADER, self.FRAG_SHADER)
        self.probe = probe
        self.__setitem__('u_x_min', self.probe.x_limits[0])
        self.__setitem__('u_x_max', self.probe.x_limits[1])
        self.__setitem__('u_y_min', self.probe.y_limits[0])
        self.__setitem__('u_y_max', self.probe.y_limits[1])
        self.__setitem__('u_d_scale', self.probe.minimum_interelectrode_distance)
        self.data = data

    @property
    def nb_data(self):
        try:
            return self.data.shape[0]
        except Exception:
            return 0

    @property
    def nb_points(self):
        try:
            return self.data.shape[1]
        except Exception:
            return 0

    @property
    def positions(self):
        return np.c_[np.repeat(self.probe.x.astype(np.float32), repeats=self.nb_points),
                     np.repeat(self.probe.y.astype(np.float32), repeats=self.nb_points)]

    @property
    def initialized(self):
        if self.nb_points > 0:
            return True
        else:
            return False

    @property
    def max_y_axis(self):
        if self.initialized:
            return np.max(self.data)
        else:
            return 0    

    def _generate_selection(self, selection=[]):
        if selection is not None and self.selection != selection:
            self.selection = selection
        data = np.zeros(self.nb_data, dtype=np.int32)
        data[self.selection] = 1
        return data

    # def on_mouse_wheel(self, event):

    #     modifiers = event.modifiers

    #     if keys.CONTROL in modifiers:
    #         dx = np.sign(event.delta[1]) * 0.01
    #         v_scale = self.programs['templates']['u_v_scale']
    #         v_scale_new = v_scale * np.exp(dx)
    #         self.programs['templates']['u_v_scale'] = v_scale_new
    #     elif keys.SHIFT in modifiers:
    #         time_ref = self._time_max
    #         dx = np.sign(event.delta[1]) * 0.01
    #         t_scale = self.programs['templates']['u_t_scale']
    #         t_scale_new = t_scale * np.exp(dx)
    #         t_scale_new = max(t_scale_new, time_ref / self._time_max)
    #         t_scale_new = min(t_scale_new, time_ref / self._time_min)
    #         self.programs['templates']['u_t_scale'] = t_scale_new
    #     else:
    #         dx = np.sign(event.delta[1]) * 0.01
    #         x_min_new = self.programs['templates']['u_x_min'] * np.exp(dx)
    #         x_max_new = self.programs['templates']['u_x_max'] * np.exp(dx)
    #         self.programs['templates']['u_x_min'] = x_min_new
    #         self.programs['templates']['u_x_max'] = x_max_new

    #         y_min_new = self.programs['templates']['u_y_min'] * np.exp(dx)
    #         y_max_new = self.programs['templates']['u_y_max'] * np.exp(dx)
    #         self.programs['templates']['u_y_min'] = y_min_new
    #         self.programs['templates']['u_y_max'] = y_max_new

    #     # # TODO emit signal to update the spin box.

    #     self.update()

    #     return

    # def on_mouse_move(self, event):

    #     if event.press_event is None:
    #         return

    #     modifiers = event.modifiers
    #     p1 = event.press_event.pos
    #     p2 = event.pos

    #     p1 = np.array(event.last_event.pos)[:2]
    #     p2 = np.array(event.pos)[:2]

    #     dx, dy = 0.1 * (p1 - p2)

    #     self.programs['templates']['u_x_min'] += dx
    #     self.programs['templates']['u_x_max'] += dx
    #     self.programs['templates']['u_y_min'] += dy
    #     self.programs['templates']['u_y_max'] += dy

    #     # # TODO emit signal to update the spin box.

    #     self.update()

    #     return

    def set_attribute(self, attribute, data):
        #set_attr(self, attribute, data)

        if attribute in ['a_selection', 'a_index_selection']:
            data = np.repeat(data, repeats=self.nb_points).astype(np.float32)
        elif attribute in ['a_colors']:
            data = np.repeat(data, repeats=self.nb_points, axis=0).astype(np.float32)
        elif attribute in ['a_values']:
            data = data.ravel().astype(np.float32)
        elif attribute in ['a_index', 'a_positions']:
            data = np.tile(data, reps=self.nb_data).astype(np.float32)
        elif attribute in ['a_sample_index']:
            data = np.tile(data, reps = self.nb_data * self.nb_channels).astype(np.float32)
        
        if attribute in ['a_index', 'a_index_selection', 'a_colors', 'a_values', 'a_sample_index', 'a_positions', 'a_selection']:
            self.__setitem__(attribute, gloo.VertexBuffer(data))
        else:
            self.__setitem__(attribute, data)

        # self.programs['templates']['a_sample_index'] = self.template_sample_index
        # self.programs['templates']['a_template_selected'] = self.template_selected
        # self.programs['templates']['u_nb_samples_per_signal'] = self.nb_samples_per_template
        
        # self.programs['templates']['u_t_scale'] = self._time_max / params['time']['init']
        # self.programs['templates']['u_v_scale'] = params['voltage']['init']

    def set_zoom_y_axis(self, factor=1):
        self.zoom[1] = factor
        self.set_attribute('u_scale', self.zoom)

    def set_x_scale(self, value):
        self.__setitem__('u_t_scale', value)

    def set_y_scale(self, value):
        self.__setitem__('u_v_scale', value)

    def set_data(self, data, colors=None):

        self.data = data
        if colors is None:
            colors = np.zeros((self.nb_data, 3), dtype=np.float32)

        self.set_attribute('a_index', np.repeat(np.arange(self.nb_channels),
                                                 repeats=self.nb_points))

        self.set_attribute('a_positons', self.positions)
        self.set_attribute('a_colors', colors)
        self.set_attribute('a_values', data)
        self.set_attribute('u_nb_points', self.nb_points)
        self.set_attribute('a_sample_index', np.arange(0, self.nb_points))
        self.set_selection(None)

    def set_selection(self, selection):
        selection = self._generate_selection(selection)
        self.set_attribute('a_selection', selection)


class SingleLinePlot(LinesPlot):

    FRAG_SHADER = """
    varying float v_xpos;
    varying float v_index_selection;
    varying vec3 v_colors;
    varying float v_selection;

    // Fragment shader.
    void main() {
        gl_FragColor = vec4(v_colors, 1.0);
        if (v_selection == 0.0)
            discard;
        if (fract(v_index_selection) > 0.0 || (v_xpos) > 0.9)
            discard;
    }
    """

    VERT_SHADER = """
    attribute float a_values;
    attribute float a_selection;
    attribute vec3 a_colors;
    attribute float a_index_selection;
    attribute float a_index_xaxis;

    uniform vec2 u_scale;
    uniform float u_max_value;
    uniform float u_nb_points;

    varying vec3 v_colors;
    varying float v_selection;
    varying float v_index_selection;
    varying float v_xpos;

    void main() {
        float x = -0.9 + (1.8 * (a_index_xaxis / u_nb_points) * u_scale.x);
        float y = -0.9 + a_values/u_max_value;
        vec2 position = vec2(x, y);   
        gl_Position = vec4(position, 0.0, 1.0);
        v_colors = a_colors;
        v_index_selection = a_index_selection;
        v_xpos = position.x;
        v_selection = a_selection;
    }
    """

    def __init__(self, data=np.zeros((0, 0), dtype=np.float32)):
        LinesPlot.__init__(self, self.VERT_SHADER, self.FRAG_SHADER)
        self.zoom = np.array([1.0, 1.0], dtype=np.float32)
        self.selection = []
        self.set_data(data)

    @property
    def nb_data(self):
        try:
            return self.data.shape[0]
        except Exception:
            return 0

    @property
    def nb_points(self):
        try:
            return self.data.shape[1]
        except Exception:
            return 0        

    @property
    def initialized(self):
        if self.nb_points > 0:
            return True
        else:
            return False

    @property
    def max_y_axis(self):
        if self.initialized:
            return np.max(self.data)
        else:
            return 0

    def _generate_selection(self, selection=[]):

        if selection is not None and self.selection != selection:
            self.selection = selection
        data = np.zeros(self.nb_data, dtype=np.int32)
        data[self.selection] = 1
        return data

    def on_mouse_wheel(self, event):

        modifiers = event.modifiers

        dx = np.sign(event.delta[1]) * 0.1
        self.set_zoom_y_axis(self.zoom[1] * np.exp(dx))
        # # TODO emit signal to update the spin box.
        self.update()

        return

    def set_attribute(self, attribute, data):
        #set_attr(self, attribute, data)

        if attribute in ['a_selection', 'a_index_selection']:
            data = np.repeat(data, repeats=self.nb_points).astype(np.float32)
        elif attribute in ['a_colors']:
            data = np.repeat(data, repeats=self.nb_points, axis=0).astype(np.float32)
        elif attribute in ['a_values']:
            data = data.ravel().astype(np.float32)
        elif attribute in ['a_index_xaxis']:
            data = np.tile(data, reps=self.nb_data).astype(np.float32)

        if attribute in ['a_selection', 'a_index_selection', 'a_colors', 'a_values', 'a_index_xaxis']:
            self.__setitem__(attribute, gloo.VertexBuffer(data))
        else:
            self.__setitem__(attribute, data)

    def set_zoom_y_axis(self, factor=1):
        self.zoom[1] = factor
        self.set_attribute('u_scale', self.zoom)

    def set_data(self, data, colors=None):

        self.data = data
        if colors is None:
            colors = np.zeros((self.nb_data, 3), dtype=np.float32)

        self.set_attribute('a_values', data)
        self.set_attribute('a_colors', colors)
        self.set_attribute('a_index_xaxis', np.arange(self.nb_points))
        self.set_attribute('a_index_selection', np.arange(self.nb_data))
        self.set_attribute('u_scale', self.zoom)
        self.set_attribute('u_max_value', self.max_y_axis)
        self.set_attribute('u_nb_points', self.nb_points)
        self.set_selection(None)

    def set_selection(self, selection):
        selection = self._generate_selection(selection)   
        self.set_attribute('a_selection', selection)
