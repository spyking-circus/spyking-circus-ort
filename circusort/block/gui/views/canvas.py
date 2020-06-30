import utils.widgets as wid
import numpy as np

from vispy import app, gloo
from vispy.util import keys
from views.programs import LinesPlot, BoxPlot, ProbeBoxPlot
from circusort.io.probe import load_probe


class ViewCanvas(app.Canvas):

    def __init__(self, probe_path, title="Vispy Canvas", box=None):

        app.Canvas.__init__(self, title=title)
        self.programs = {}
        self.probe = load_probe(probe_path)
        self.controler = None

        gloo.set_viewport(0, 0, *self.physical_size)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

        if box == 'single':
            self.add_curve('box', BoxPlot())
        elif box == 'multi':
            self.add_curve('box', ProbeBoxPlot(self.probe))

    def add_curve(self, name, plot):
        self.programs[name] = plot

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)
        return

    def get_colors(self, nb_templates, seed=42):
        np.random.seed(seed)
        return np.random.uniform(size=(nb_templates, 3), low=0.3, high=.9).astype(np.float32)

    def on_draw(self, event):

        _ = event
        gloo.clear()
        gloo.set_viewport(0, 0, *self.physical_size)
        for p in self.programs.values():
            p._draw()
        self.update()
        return

    # def on_mouse_move(self, event):
        # for p in self.programs.values():
        #     p._on_mouse_move(event)
        # self.update()

    # def on_mouse_wheel(self, event):
        # for p in self.programs.values():
        #     p.on_mouse_wheel(event)
        # self.update()

    def on_reception(self, data):
        self._on_reception(data)
        self.update()

    def set_value(self, dictionary):
        for key, value in dictionary.items():
            self._set_value(key, value)
        self.update()

    def highlight_selection(self, selection):
        self._highlight_selection(selection)
        self.update()
        return