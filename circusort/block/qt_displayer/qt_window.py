# coding=utf-8
from PyQt4.QtCore import Qt
from PyQt4.QtGui import *

from circusort.block.qt_displayer.qt_canvas import VispyCanvas
from circusort.block.qt_displayer.qt_thread import QtThread


class QtWindow(QMainWindow):

    def __init__(self, params_pipe, number_pipe, data_pipe, probe_path=None, screen_resolution=None):

        QMainWindow.__init__(self)

        self._canvas = VispyCanvas(probe_path=probe_path)
        central_widget = self._canvas.native

        # Create controls widgets.
        label_time = QLabel()
        label_time.setText(u"time")
        label_voltage = QLabel()
        label_voltage.setText(u"voltage")
        dsp_time = QDoubleSpinBox()
        dsp_voltage = QDoubleSpinBox()
        label_time_unit = QLabel()
        label_time_unit.setText(u"ms")
        label_voltage_unit = QLabel()
        label_voltage_unit.setText(u"µV")
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        # Create controls grid.
        grid = QGridLayout()
        grid.addWidget(label_time, 0, 0)
        grid.addWidget(dsp_time, 0, 1)
        grid.addWidget(label_time_unit, 0, 2)
        grid.addWidget(label_voltage, 1, 0)
        grid.addWidget(dsp_voltage, 1, 1)
        grid.addWidget(label_voltage_unit, 1, 2)
        grid.addItem(spacer)

        # Create controls group.
        controls_group = QGroupBox()
        controls_group.setLayout(grid)

        # Create controls dock.
        dock = QDockWidget()
        dock.setWidget(controls_group)
        dock.setWindowTitle("Controls")

        # Create info widgets.
        label_time = QLabel()
        label_time.setText(u"time")
        self._label_time_value = QLabel()
        self._label_time_value.setText(u"0")
        label_time_unit = QLabel()
        label_time_unit.setText(u"s")
        info_buffer_label = QLabel()
        info_buffer_label.setText(u"buffer")
        self._info_buffer_value_label = QLabel()
        self._info_buffer_value_label.setText(u"0")
        info_buffer_unit_label = QLabel()
        info_buffer_unit_label.setText(u"")
        info_probe_label = QLabel()
        info_probe_label.setText(u"probe")
        info_probe_value_label = QLabel()
        info_probe_value_label.setText(u"{}".format(probe_path))
        # TODO place the following info in another grid.
        info_probe_unit_label = QLabel()
        info_probe_unit_label.setText(u"")

        info_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        # Create info grid.
        info_grid = QGridLayout()
        # # Time row.
        info_grid.addWidget(label_time, 0, 0)
        info_grid.addWidget(self._label_time_value, 0, 1)
        info_grid.addWidget(label_time_unit, 0, 2)
        # # Buffer row.
        info_grid.addWidget(info_buffer_label, 1, 0)
        info_grid.addWidget(self._info_buffer_value_label, 1, 1)
        info_grid.addWidget(info_buffer_unit_label, 1, 2)
        # # Probe row.
        info_grid.addWidget(info_probe_label, 2, 0)
        info_grid.addWidget(info_probe_value_label, 2, 1)
        info_grid.addWidget(info_probe_unit_label, 2, 2)
        # # Spacer.
        info_grid.addItem(info_spacer)

        # Create info group.
        info_group = QGroupBox()
        info_group.setLayout(info_grid)

        # Create info dock.
        info_dock = QDockWidget()
        info_dock.setWidget(info_group)
        info_dock.setWindowTitle("Info")

        # Receive parameters.
        params = params_pipe[0].recv()
        self._nb_samples = params['nb_samples']
        self._sampling_rate = params['sampling_rate']

        # Create thread.
        thread = QtThread(number_pipe, data_pipe)
        thread.number_signal.connect(self.number_callback)
        thread.data_signal.connect(self.data_callback)
        thread.start()

        # Add dockable windows.
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, info_dock)
        # Set central widget.
        self.setCentralWidget(central_widget)
        # Set window size.
        if screen_resolution is not None:
            screen_width = screen_resolution.width()
            screen_height = screen_resolution.height()
            self.resize(screen_width, screen_height)
        # Set window title.
        self.setWindowTitle("SpyKING Circus ORT - Read 'n' display (Qt)")

    def number_callback(self, number):

        text = u"{}".format(number)
        self._info_buffer_value_label.setText(text)

        text = u"{:8.3f}".format(float(number) * float(self._nb_samples) / self._sampling_rate)
        self._label_time_value.setText(text)

        return

    def data_callback(self, data):

        self._canvas.update_data(data)

        return
