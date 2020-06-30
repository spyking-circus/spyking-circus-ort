import numpy as np
import scipy.signal

from vispy import app, gloo
from vispy.util import keys

from circusort.io.probe import load_probe
from circusort.io.template import load_template_from_dict

from utils.widgets import Controler

from views.canvas import ViewCanvas
from views.programs import SingleLinePlot, LinesPlot, BoxPlot
from circusort.obj.cells import Cells
from circusort.obj.cell import Cell
from circusort.obj.train import Train
from circusort.obj.amplitude import Amplitude


class ISICanvas(ViewCanvas):

    requires = ['spikes', 'time']
    name = "ISIs"

    def __init__(self, probe_path=None, params=None):
        ViewCanvas.__init__(self, probe_path, title="ISI view", box='single')
        self.cells = Cells({})
        self.programs['isis'] = SingleLinePlot()
        self.controler = ISIControler(self)

    @property
    def nb_templates(self):
        return len(self.cells)

    def zoom(self, zoom_value):
        self.programs['isis'].set_zoom_y_axis(zoom_value)
        self.update()
        return

    def _highlight_selection(self, selection):
        self.programs['isis'].set_selection(selection)
        return

    def _on_reception(self, data):
        
        spikes = data['spikes'] if 'spikes' in data else None
        self.time = data['time'] if 'time' in data else None
        old_size = self.nb_templates

        if spikes is not None:

            is_known = np.in1d(np.unique(spikes['templates']), self.cells.ids)
            not_kwown = is_known[is_known == False]

            for i in range(len(not_kwown)):
                template = None
                new_cell = Cell(template, Train([], t_min=0), Amplitude([], [], t_min=0))
                self.cells.append(new_cell)

            self.cells.add_spikes(spikes['spike_times'], spikes['amplitudes'], spikes['templates'])    
        
        self.cells.set_t_max(self.time)
        self.cells.set_t_min(0)

        isis = self.cells.interspike_interval_histogram(self.controler.bin_size, self.controler.max_time)
        isis = np.array([isi[0] for isi in isis.values()])
        
        colors = self.get_colors(self.nb_templates)
        self.programs['isis'].set_data(isis, colors)

        return

class ISIControler(Controler):

    def __init__(self, canvas, bin_size=0.005, max_time=1):
        '''
        Control widgets:
        '''

        # TODO ISI

        Controler.__init__(self, canvas)
        self.bin_size = bin_size 
        self.max_time = max_time

        self.dsb_bin_size = self.double_spin_box(label='Bin Size', unit='seconds', min_value=0.001,
                                                 max_value=1, step=0.001, init_value=self.bin_size)

        self.dsb_zoom = self.double_spin_box(label='Zoom', min_value=1, max_value=50, step=0.1,
                                             init_value=1)

        self.dsb_time_window = self.double_spin_box(label='Max time', unit='seconds',
                                                    min_value=0.1, max_value=50, step=0.1,
                                                    init_value=self.max_time)

        self.add_widget(self.dsb_bin_size, self._on_binsize_changed)
        self.add_widget(self.dsb_zoom, self._on_zoom_changed)
        self.add_widget(self.dsb_time_window, self._on_time_changed)

    def _on_binsize_changed(self, bin_size):
        self.bin_size = self.dsb_bin_size['widget'].value()
        return

    def _on_zoom_changed(self):
        zoom_value = self.dsb_zoom['widget'].value()
        self.canvas.zoom(zoom_value)
        return

    def _on_time_changed(self):
        self.max_time = self.dsb_time_window['widget'].value()
        return