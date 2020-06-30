import numpy as np

from vispy import app, gloo
from vispy.util import keys

from circusort.io.probe import load_probe
from circusort.io.template import load_template_from_dict
from circusort.obj.cells import Cells
from circusort.obj.cell import Cell
from circusort.obj.train import Train
from circusort.obj.amplitude import Amplitude

from utils.widgets import Controler
from views.canvas import ViewCanvas
from views.programs import LinesPlot, ProbeSingleLinePlot

import sys
import matplotlib.pyplot as plt


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

        self.programs['templates'] = SingleLinePlot(self.probe)
        self.cells = Cells({})

        # Final details.
        self.controler = TemplateControler(self, params)
        self.programs['templates'].set_x_scale(self._time_max / params['time']['init'])
        self.programs['templates'].set_y_scale(params['voltage']['init'])

    @property
    def nb_templates(self):
        return len(self.cells)

    # TODO : Warning always called
    def _on_reception(self, data):

        templates = data['templates'] if 'templates' in data else None
        
        if templates is not None:

            for i in range(len(templates)):
                template = load_template_from_dict(templates[i], self.probe)
                new_cell = Cell(template, Train([], t_min=0), Amplitude([], [], t_min=0))
                self.cells.append(new_cell)

            colors = self.get_colors(self.nb_templates)
            self.programs['templates'].set_data(self.cells.get_templates(), colors)

        return

    def _set_value(self, key, value):

        if key == "time":
            t_scale = self._time_max / value
            self.programs['templates'].set_x_scale(t_scale)
        elif key == "voltage":
            self.programs['templates'].set_y_scale(value)
        elif key == "templates":
            self.templates = value

    def _highlight_selection(self, selection):
        self.programs['templates'].set_selection(selection)
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
