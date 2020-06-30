# coding=utf-8
try:
    # Python 2 compatibility.
    from PyQt4.QtCore import Qt
    from PyQt4.QtGui import QMainWindow, QLabel, QDoubleSpinBox, QSpacerItem, \
        QSizePolicy, QGroupBox, QGridLayout, QLineEdit, QDockWidget, QListWidget, \
        QListWidgetItem, QAbstractItemView, QCheckBox, QTableWidget, QTableWidgetItem
except ImportError:  # i.e. ModuleNotFoundError
    # Python 3 compatibility.
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QMainWindow, QLabel, QDoubleSpinBox, QSpacerItem, \
        QSizePolicy, QGroupBox, QGridLayout, QLineEdit, QDockWidget, QListWidget, \
        QListWidgetItem, QAbstractItemView, QCheckBox, QTableWidget, QTableWidgetItem, QAction

import circusort.block.gui.utils.widgets as wid
from circusort.block.gui.utils.widgets import Controler

from circusort.block.gui.views.traces import TraceCanvas
from circusort.block.gui.views.templates import TemplateCanvas
from circusort.block.gui.views.electrodes import MEACanvas
from circusort.block.gui.views.rates import RateCanvas
from circusort.block.gui.views.isis import ISICanvas

from circusort.block.gui.thread import ThreadORT
from circusort.io.probe import load_probe
from circusort.io.template import load_template_from_dict
import numpy as np

from circusort.obj.cells import Cells
from circusort.obj.cell import Cell
from circusort.obj.train import Train
from circusort.obj.amplitude import Amplitude


_all_views_ = [TemplateCanvas]

class InfoController(Controler):

    def __init__(self, probe_path):

        Controler.__init__(self)
        self.probe_path = probe_path

        self._info_time = self.line_edit(label='Time', init_value='0', read_only=True, label_unit='s')
        self._info_buffer = self.line_edit(label='Buffer', init_value='0', read_only=True, label_unit=None)
        self._info_probe = self.line_edit(label='Probe', init_value="{}".format(self.probe_path),
                                          read_only=True, label_unit=None)

        self.add_widget(self._info_time)
        self.add_widget(self._info_buffer)
        self.add_widget(self._info_probe)


class GUIWindow(QMainWindow):

    def __init__(self, all_queues, screen_resolution=None):

        QMainWindow.__init__(self)

        self.setDockOptions(QMainWindow.AllowTabbedDocks | QMainWindow.AllowNestedDocks | QMainWindow.VerticalTabs)
        self.setDockNestingEnabled(True)
        self.all_queues = all_queues

        # Receive parameters.
        params = self.all_queues['params'].get()
        self.probe_path = params['probe_path']
        self.probe = load_probe(self.probe_path)
        self._nb_samples = params['nb_samples']
        self._sampling_rate = params['sampling_rate']
        self.real_time = int(1000*self._nb_samples / self._sampling_rate)

        self._display_list = []

        self._params = {
            'nb_samples': self._nb_samples,
            'sampling_rate': self._sampling_rate,
            'time': {
                'min': 10.0,  # ms
                'max': 100.0,  # ms
                'init': 100.0,  # ms
            },
            'voltage': {
                'min': -100,  # µV
                'max': 100,  # µV
                'init': 50.0,  # µV
            },
            'templates': self._display_list
        }

        self.cells = Cells({})
        self._nb_buffer = 0

        # Load the  canvas
        self._canvas_loading()
        self._control_loading()
        self.menu_mw()

        # Load the dock widget
        self.info_controler = InfoController(self.probe_path)
        self.addDockWidget(Qt.TopDockWidgetArea, self.info_controler.dock_control('Info'), Qt.Horizontal)

        
        # TODO create a TableWidget method

        self._selection_templates = QTableWidget()
        self._selection_templates.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self._selection_templates.setColumnCount(3)
        self._selection_templates.setVerticalHeaderLabels(['Nb template', 'Channel', 'Amplitude'])
        self._selection_templates.insertRow(0)
        self._selection_templates.setItem(0, 0, QTableWidgetItem('Nb template'))
        self._selection_templates.setItem(0, 1, QTableWidgetItem('Channel'))
        self._selection_templates.setItem(0, 2, QTableWidgetItem('Amplitude'))

        # Create info grid.
        templates_grid = QGridLayout()
        # # Add Channel selection
        # grid.addWidget(label_selection, 3, 0)
        templates_grid.addWidget(self._selection_templates, 0, 1)

        def add_template():
            items = self._selection_templates.selectedItems()
            self._display_list = []
            for i in range(len(items)):
                self._display_list.append(i)
            self._on_templates_changed()

        # self._selection_templates.itemClicked.connect(add_template)

        # Template selection signals
        self._selection_templates.itemSelectionChanged.connect(self.selected_templates)

        # self._selection_templates.itemPressed(0, 1).connect(self.sort_template())

        # # Add spacer.
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        templates_grid.addItem(spacer)

        # Create controls group.
        templates_group = QGroupBox()
        templates_group.setLayout(templates_grid)

        # # Create controls dock.
        templates_dock = QDockWidget()
        templates_dock.setWidget(templates_group)
        templates_dock.setWindowTitle("Channels selection")
        self.addDockWidget(Qt.TopDockWidgetArea, templates_dock, Qt.Horizontal)

        # Create thread.
        thread2 = ThreadORT(self.all_queues, self.real_time)
        thread2.reception_signal.connect(self._reception_callback)
        thread2.start()

        # self.setCentralWidget(QLineEdit())

        # Set window size.
        if screen_resolution is not None:
            screen_width = screen_resolution.width()
            screen_height = screen_resolution.height()
            self.resize(screen_width, screen_height)

        # Set window title.
        self.setWindowTitle("SpyKING Circus ORT - Read 'n' Qt display")
        self.show()

    @property
    def nb_templates(self):
        return len(self.cells)

    # -----------------------------------------------------------------------------
    # Canvas & Control handling
    # -----------------------------------------------------------------------------

    def _canvas_loading(self):
        """ Load the vispy canvas from the files """

        self.all_canvas = {}
        self.all_docks = {}
        for count, view in enumerate(_all_views_):
            label = view.name
            self.all_canvas[label] = view(probe_path=self.probe_path, params=self._params)
            self.all_docks[label] = wid.dock_canvas(self.all_canvas[label], label)
            if np.mod(count, 2) == 0:
                position = Qt.LeftDockWidgetArea
            else:
                position = Qt.RightDockWidgetArea
            self.addDockWidget(position, self.all_docks[label])
        
    def _control_loading(self):
        """ """
        self.all_controls = {}
        for view in self.all_canvas.values():
            if view.controler is not None:
                label = view.name                
                self.all_controls[label] = view.controler.dock_control()
                self.addDockWidget(Qt.TopDockWidgetArea, self.all_controls[label], Qt.Horizontal)

    # -----------------------------------------------------------------------------
    # Menu Creation
    # -----------------------------------------------------------------------------

    def menu_mw(self):
        """ Menu """
        main_menu = self.menuBar()
        main_menu.setNativeMenuBar(False)  # Disables the native menu bar on Mac

        file_menu = main_menu.addMenu("File")
        edit_menu = main_menu.addMenu("Edit")
        view_menu = main_menu.addMenu("Views")
        help_menu = main_menu.addMenu("Help")

        self.all_views = {}
        self.all_toggles = {}
        for view in self.all_canvas.values():
            label = view.name
            self.all_views[label] = QAction(label, self)
            if view.controler is None:
                self.all_toggles[label] = self.all_docks[label].toggleViewAction()
            elif view.controler is not None:  
                self.all_toggles[label] = QAction(label, self)
                self.all_toggles[label].setCheckable(True)
                self.all_toggles[label].setChecked(True)
                self.all_toggles[label].changed.connect(lambda: self._visibility(self.all_toggles[label].isChecked(),
                                                                         self.all_docks[label],
                                                                         self.all_controls[label]))

            view_menu.addAction(self.all_toggles[label])

    def _visibility(self, state, canvas, control):
        canvas.setVisible(state)
        control.setVisible(state)

        return

    # -----------------------------------------------------------------------------
    # Data handling
    # -----------------------------------------------------------------------------

    def _number_callback(self, number):

        self._nb_buffer = float(number)
        nb_buffer = u"{}".format(number)
        self.info_controler._info_buffer['widget'].setText(nb_buffer)

        txt_time = u"{:8.3f}".format(self.time)
        self.info_controler._info_time['widget'].setText(txt_time)

        return

    def _reception_callback(self, data):
        

        templates = data['templates'] if 'templates' in data else None
        spikes = data['spikes'] if 'spikes' in data else None
        self.time = self._nb_samples * (self._nb_buffer + 1) / self._sampling_rate

        if 'number' in data:
            self._number_callback(data['number'])

        self.new_templates = []
        
        if templates is not None:
            for i in range(len(templates)):
                mask = spikes['templates'] == i
                template = load_template_from_dict(templates[i], self.probe)
                self.new_templates += [template]

                new_cell = Cell(template, Train([]), Amplitude([], []))
                self.cells.append(new_cell)
                self._selection_templates.insertRow(self.nb_templates)

                channel = template.channel
                amplitude = template.peak_amplitude()
                self._selection_templates.setItem(self.nb_templates, 0, QTableWidgetItem(str(self.nb_templates)))
                self._selection_templates.setItem(self.nb_templates, 1, QTableWidgetItem(str(channel)))
                self._selection_templates.setItem(self.nb_templates, 2, QTableWidgetItem(str(amplitude)))

        for canvas in self.all_canvas.values():
            to_send = self.prepare_data(canvas, data)
            canvas.on_reception(to_send)

        return

    def prepare_data(self, canvas, data):

        to_get = canvas.requires
        to_send = {}

        for key in to_get:
            if key == 'nb_templates':
                to_send[key] = self.nb_templates
            elif key == 'barycenters':
                to_send[key] = [t.center_of_mass(self.probe) for t in self.new_templates]
            elif key in ['data', 'peaks', 'thresholds', 'templates', 'spikes']:
                to_send[key] = data[key]
            elif key == 'time':
                to_send[key] = self.time

        return to_send

    def selected_templates(self):
        list_templates = []

        for i in range(self.nb_templates):
            if self._selection_templates.item(i+1, 0).isSelected() and \
                    self._selection_templates.item(i+1, 1).isSelected() and \
                    self._selection_templates.item(i+1, 2).isSelected():
                list_templates.append(i)

        for canvas in self.all_canvas.values():
            canvas.highlight_selection(list_templates)
    
        return
