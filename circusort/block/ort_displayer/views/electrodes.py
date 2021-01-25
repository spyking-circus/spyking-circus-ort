import numpy as np
import math

from vispy import app, gloo
from vispy.util import keys

from circusort.io.probe import load_probe
from circusort.io.template import load_template_from_dict

from circusort.block.ort_displayer.views.canvas import ViewCanvas
from circusort.block.ort_displayer.views.programs import LinesPlot, ScatterPlot

BOUNDARY_VERT_SHADER = """
// Coordinates of the position of the box.
attribute vec2 a_pos_probe;
// Coordinates of the position of the corner.
attribute vec2 a_corner_position;
// Uniform variables used to transform the subplots.
uniform float u_x_min;
uniform float u_x_max;
uniform float u_y_min;
uniform float u_y_max;
uniform float u_d_scale;
// 2D scaling factors + dragging (zooming).
uniform vec2 u_scale;
uniform vec2 u_pan;


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
    vec2 b = vec2(-1.0 + 2.0 * (a_pos_probe.x - u_x_min) / w,
                    -1.0 + 2.0 * (a_pos_probe.y - u_y_min) / h);
    // Apply the transformation.
    gl_Position = vec4(a *u_scale* (p + u_pan) + b, 0.0, 1.0);
}
"""

CHANNELS_VERT_SHADER = """
//uniform vec2 resolution;
//attribute vec2 channel_centers;
attribute vec2 a_channel_position;
attribute float a_selected_channel;

uniform float u_x_min;
uniform float u_x_max;
uniform float u_y_min;
uniform float u_y_max;
uniform float u_d_scale;
uniform float radius;
// 2D scaling factor (zooming).
uniform vec2 u_scale;
uniform vec2 u_pan;

varying vec2 v_center;
varying float v_radius; 
varying float v_select;

varying vec4 v_fg_color;
varying vec4 v_selec_color;
varying vec4 v_unsel_color;
varying float v_linewidth;
varying float v_antialias;

void main(){

    float w = u_x_max - u_x_min;
    float h = u_y_max - u_y_min;
    //vec2 p = (0.0, 0.0);
    vec2 a = vec2(1.0 / (1.0 + w / u_d_scale), 1.0 / (1.0 + h / u_d_scale));
    vec2 b = vec2(-1.0 + 2.0 * (a_channel_position.x - u_x_min) / w, -1.0 + 2.0 * (a_channel_position.y - u_y_min) / h);
    //center = vec2(a * p + b);
    vec2 center = b;
    v_center = center;
    v_select = a_selected_channel;
    v_radius = radius;
    
    v_linewidth = 1.0;
    v_antialias = 1.0;
    v_fg_color  = vec4(1.0,1.0,1.0,0.5);
    v_selec_color = vec4(0.7, 0.7, 0.7, 1.0);
    v_unsel_color = vec4(0.3, 0.3, 0.3, 1.0);
    
    gl_PointSize = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
    gl_Position = vec4((center+u_pan) * u_scale , 0.0, 1.0);
}
"""

BARYCENTER_VERT_SHADER = """
attribute vec2 a_barycenter_position;
attribute float a_selected_template;
attribute vec3 a_color;

uniform float u_x_min;
uniform float u_x_max;
uniform float u_y_min;
uniform float u_y_max;
uniform float u_d_scale;
uniform float radius;
// 2D scaling factor (zooming).
uniform vec2 u_scale;
uniform vec2 u_pan;

varying float v_radius; 
varying float v_selected_temp;

varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_linewidth;
varying float v_antialias;

void main() {
    float w = u_x_max - u_x_min;
    float h = u_y_max - u_y_min;
    vec2 a_bis = vec2(w , h);
    vec2 p = a_barycenter_position;
    vec2 a = vec2(1.0 / (1.0 + w / u_d_scale), 1.0 / (1.0 + h / u_d_scale));
    //gl_PointSize = 2.0 + ceil(2.0*radius);
    //gl_PointSize  = radius;
    //TODO modify the following with parameters
    gl_Position = vec4((p/135 + u_pan) * u_scale, 0.0, 1.0);
    
    v_linewidth = 1.0;
    v_antialias = 1.0;
    v_fg_color  = vec4(1.0, 1.0, 1.0, 0.5);
    v_bg_color  = vec4(a_color,    1.0);
    gl_PointSize = 2.0*(radius + v_linewidth + 1.5*v_antialias);    
      
    v_selected_temp = a_selected_template;
    v_radius = radius;
}
"""

BOUNDARY_FRAG_SHADER = """
// Fragment shader.
void main() {
    gl_FragColor = vec4(0.25, 0.25, 0.25, 1.0);
}
"""

CHANNELS_FRAG_SHADER = """
varying vec2 v_center;
varying float v_radius;
varying float v_select;

varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_linewidth;
varying float v_antialias;
varying vec4 v_selec_color;
varying vec4 v_unsel_color;
// Fragment shader.
void main() {
    float size = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
    float t = v_linewidth/2.0-v_antialias;
    float r = length((gl_PointCoord.xy - vec2(0.5,0.5))*size);
    float d = abs(r - v_radius) - t;
    if( d < 0.0 )
        gl_FragColor = v_fg_color;
    else
    {
        float alpha = d/v_antialias;
        alpha = exp(-alpha*alpha);
        if (r > v_radius)
            gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
        else
        {
            //gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
            if (v_select == 1)
                gl_FragColor = vec4(0.1, 1.0, 0.1, alpha);
            else
                gl_FragColor = vec4(0.3, 0.3, 0.3, alpha);        
        }
    }
}
"""

BARYCENTER_FRAG_SHADER = """
varying float v_radius;
varying float v_selected_temp;

varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_linewidth;
varying float v_antialias;

// Fragment shader.
void main() {
    float size = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
    float t = v_linewidth/2.0-v_antialias;
    float r = length((gl_PointCoord.xy - vec2(0.5,0.5))*size);
    float d = abs(r - v_radius) - t;
    
    if (v_selected_temp == 0.0)
        discard;
    else 
    {
        if( d < 0.0 )
            gl_FragColor = v_fg_color;
        else
        {
            float alpha = d/v_antialias;
            alpha = exp(-alpha*alpha);
            if (r > v_radius)
                gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
            else
                gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
        }
    }
}
"""



class MEACanvas(ViewCanvas):

    requires = ['barycenters']
    name = "Electrodes"

    def __init__(self, probe_path=None, params=None):
        ViewCanvas.__init__(self, probe_path, title="Probe view")

        # self.channels = params['channels']
        self.nb_channels = self.probe.nb_channels
        self.initialized = False

        # TODO Add method to probe file to extract minimum coordinates without the interelectrode dist
        x_min, x_max = self.probe.x_limits[0] + self.probe.minimum_interelectrode_distance,\
                       self.probe.x_limits[1] - self.probe.minimum_interelectrode_distance
        y_min, y_max = self.probe.y_limits[0] + self.probe.minimum_interelectrode_distance, \
                       self.probe.y_limits[1] - self.probe.minimum_interelectrode_distance

        probe_corner = np.array([[x_max, y_max],
                                 [x_min, y_max],
                                 [x_min, y_min],
                                 [x_max, y_min],
                                 [x_max, y_max]], dtype=np.float32)

        corner_bound_positions = np.array([[+1.0, +1.0],
                                           [-1.0, +1.0],
                                           [-1.0, -1.0],
                                           [+1.0, -1.0],
                                           [+1.0, +1.0]], dtype=np.float32)


        # Define GLSL program.
        self.programs['boundary'] = LinesPlot(BOUNDARY_VERT_SHADER, BOUNDARY_FRAG_SHADER)
        self.programs['boundary']['a_pos_probe'] = probe_corner
        self.programs['boundary']['a_corner_position'] = corner_bound_positions
        self.programs['boundary']['u_x_min'] = self.probe.x_limits[0]
        self.programs['boundary']['u_x_max'] = self.probe.x_limits[1]
        self.programs['boundary']['u_y_min'] = self.probe.y_limits[0]
        self.programs['boundary']['u_y_max'] = self.probe.y_limits[1]
        self.programs['boundary']['u_d_scale'] = self.probe.minimum_interelectrode_distance
        self.programs['boundary']['u_scale'] = (1.0, 1.0)
        self.programs['boundary']['u_pan'] = (0.0, 0.0)

        # Probe
        channel_pos = np.c_[
            np.repeat(self.probe.x.astype(np.float32), repeats=1),
            np.repeat(self.probe.y.astype(np.float32), repeats=1),
        ]
        selected_channels = np.ones(self.nb_channels, dtype=np.float32)

        self.programs['channels'] = ScatterPlot(CHANNELS_VERT_SHADER, CHANNELS_FRAG_SHADER)
        self.programs['channels']['a_channel_position'] = channel_pos
        self.programs['channels']['a_selected_channel'] = selected_channels
        self.programs['channels']['radius'] = 10
        self.programs['channels']['u_x_min'] = self.probe.x_limits[0]
        self.programs['channels']['u_x_max'] = self.probe.x_limits[1]
        self.programs['channels']['u_y_min'] = self.probe.y_limits[0]
        self.programs['channels']['u_y_max'] = self.probe.y_limits[1]
        self.programs['channels']['u_scale'] = (1.0, 1.0)
        self.programs['channels']['u_pan'] = (0.0, 0.0)
        self.programs['channels']['u_d_scale'] = self.probe.minimum_interelectrode_distance
        #self.programs['channels']['u_d_scale'] = self.probe.minimum_interelectrode_distance

        #Barycenters
        self.nb_templates = 0
        self.selected_bary = 0
        barycenter_position = np.zeros((self.nb_templates, 2), dtype=np.float32)
        temp_selected = np.ones(self.nb_templates, dtype=np.float32)
        self.barycenter = np.zeros((self.nb_templates, 2), dtype=np.float32)
        self.list_selected_templates = []
        
        self.bary_color = self.get_colors(self.nb_templates)

        self.programs['barycenters'] = ScatterPlot(BARYCENTER_VERT_SHADER, BARYCENTER_FRAG_SHADER)
        self.programs['barycenters']['a_barycenter_position'] = self.barycenter
        self.programs['barycenters']['a_selected_template'] = temp_selected
        self.programs['barycenters']['a_color'] = self.bary_color
        self.programs['barycenters']['radius']  = 5
        self.programs['barycenters']['u_x_min'] = self.probe.x_limits[0]
        self.programs['barycenters']['u_x_max'] = self.probe.x_limits[1]
        self.programs['barycenters']['u_y_min'] = self.probe.y_limits[0]
        self.programs['barycenters']['u_y_max'] = self.probe.y_limits[1]
        self.programs['barycenters']['u_scale'] = (1.0, 1.0)
        self.programs['barycenters']['u_pan'] = (0.0, 0.0)
        self.programs['barycenters']['u_d_scale'] = self.probe.minimum_interelectrode_distance

    def _normalize(self, x_y):
        x, y = x_y
        w, h = float(self.size[0]), float(self.size[1])
        return x / (w / 2.) - 1., y / (h / 2.) - 1.

    def on_mouse_wheel(self, event):
        dx = np.sign(event.delta[1]) * .05
        scale_x, scale_y = self.programs['boundary']['u_scale']
        scale_x_new, scale_y_new = (scale_x * math.exp(2.5 * dx),
                                    scale_y * math.exp(2.5 * dx))
        new_scale = (max(1, scale_x_new), max(1, scale_y_new))
        self.programs['boundary']['u_scale'] = new_scale
        self.programs['channels']['u_scale'] = new_scale
        self.programs['barycenters']['u_scale'] = new_scale
        self.update()

    def on_mouse_move(self, event):
        if event.is_dragging:
            x0, y0 = self._normalize(event.press_event.pos)
            x1, y1 = self._normalize(event.last_event.pos)
            x, y = self._normalize(event.pos)
            dx, dy = x - x1, -(y - y1)
            button = event.press_event.button

            pan_x, pan_y = self.programs['boundary']['u_pan']
            scale_x, scale_y = self.programs['boundary']['u_scale']

            if button == 1:
                self.programs['boundary']['u_pan'] = (pan_x+dx/scale_x, pan_y+dy/scale_y)
                self.programs['channels']['u_pan'] = (pan_x + dx / scale_x, pan_y + dy / scale_y)
                self.programs['barycenters']['u_pan'] = (pan_x + dx / scale_x, pan_y + dy / scale_y)
            elif button == 2:
                scale_x_new, scale_y_new = (scale_x * math.exp(2.5*dx),
                                            scale_y * math.exp(2.5*dy))
                self.programs['boundary']['u_scale'] = (scale_x_new, scale_y_new)
                self.programs['barycenters']['u_scale'] = (scale_x_new, scale_y_new)
                self.programs['channels']['u_scale'] = (scale_x_new, scale_y_new)

                self.programs['boundary']['u_pan'] = (pan_x -
                                         x0 * (1./scale_x - 1./scale_x_new),
                                         pan_y +
                                         y0 * (1./scale_y - 1./scale_y_new))
                self.programs['barycenters']['u_pan'] = (pan_x -
                                         x0 * (1. / scale_x - 1. / scale_x_new),
                                         pan_y +
                                         y0 * (1. / scale_y - 1. / scale_y_new))
                self.programs['channels']['u_pan'] = (pan_x -
                                         x0 * (1. / scale_x - 1. / scale_x_new),
                                         pan_y +
                                         y0 * (1. / scale_y - 1. / scale_y_new))
            self.update()

    # def selected_channels(self, L):
    #     channels_selected = np.zeros(self.nb_channels, dtype=np.float32)
    #     # Remove redundant channels
    #     for i in set(L):
    #         channels_selected[i] = 1
    #     self.programs['channels']['a_selected_channel'] = channels_selected
    #     self.update()
    #     return

    def _highlight_selection(self, selection):
        self.list_selected_templates = [0] * self.nb_templates
        for i in selection:
            self.list_selected_templates[i] = 1
        self.selected_bary = np.array(self.list_selected_templates, dtype=np.float32)
        self.programs['barycenters']['a_selected_template'] = self.selected_bary
        return

    def _on_reception(self, data):
        
        bar = data['barycenters'] if 'barycenters' in data else None
        if bar is not None:
            for b in bar:
                self.barycenter = np.vstack((self.barycenter, np.array(b, dtype=np.float32)))
                self.list_selected_templates.append(0)
                self.nb_templates += 1

            self.selected_bary = np.array(self.list_selected_templates, dtype=np.float32)

            self.programs['barycenters']['a_barycenter_position'] = self.barycenter
            self.programs['barycenters']['a_selected_template'] = self.selected_bary
            self.programs['barycenters']['a_color'] = self.get_colors(self.nb_templates)
        return



