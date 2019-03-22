import sys
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from core.gui_utils import center, validate_session, Communicate
from core.default_parameters import project_name, default_filename, defaultXAxis, defaultYAxis, defaultZAxis
from core.Tint_Matlab import find_tet
from core.plot_functions import plot_features
import os
import json
import time


class MainWindow(QtWidgets.QWidget):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("%s - Main Window" % project_name)  # sets the main window title

        # initializing attributes
        self.win = None
        self.quit_btn = None
        self.filename = None
        self.choose_filename_btn = None
        self.x_axis_cb = None
        self.y_axis_cb = None
        self.z_axis_cb = None
        self.tetrode_cb = None
        self.choice = None
        self.feature_plot = None

        self.LogError = Communicate()
        self.LogError.signal.connect(self.raiseError)

        # various setting files
        project_dir = os.path.dirname(os.path.abspath("__file__"))
        if os.path.basename(project_dir) != project_name:
            project_dir = os.path.dirname(sys.argv[0])

        # defining the directory filepath
        self.PROJECT_DIR = project_dir  # project directory
        self.SETTINGS_DIR = os.path.join(self.PROJECT_DIR, 'settings')  # settings directory
        self.settings_filename = os.path.join(self.SETTINGS_DIR, 'settings.json')

        self.settings = self.get_settings()

        # create Settings Directory
        if not os.path.exists(self.SETTINGS_DIR):
            os.mkdir(self.SETTINGS_DIR)

        # create settings file
        if not os.path.exists(self.settings_filename):
            with open(self.settings_filename, 'w') as f:
                json.dump({}, f)

        # setting PyQtGraph settings
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.initialize()  # initializes the main window

    def initialize(self):
        """
        This method will initialize the Main Window, create all the widgets and what not
        :return:
        """

        # ---------------- session filename options ------------------------------------------

        filename_layout = QtWidgets.QHBoxLayout()

        filename_label = QtWidgets.QLabel("Filename:")
        self.filename = QtWidgets.QLineEdit()
        self.filename.setText(default_filename)
        self.filename.textChanged.connect(self.filename_changed)
        self.filename.setToolTip('The name of the session that you will analyze in gebaSpike!')
        self.choose_filename_btn = QtWidgets.QPushButton("Choose Filename")
        self.choose_filename_btn.setToolTip('This button will allow you to choose the filename to analyze!')
        self.choose_filename_btn.clicked.connect(self.choose_filename)
        for widget in [self.choose_filename_btn, filename_label, self.filename]:
            filename_layout.addWidget(widget)

        # -------- spike parameter options -------------------------------------

        feature_options = ['None', 'PC1', 'PC2', 'PC3', 'PC4', 'A', 'P', 'T', 'tT', 'tP', 'E']

        spike_parameter_layout = QtWidgets.QHBoxLayout()
        x_axis_label = QtWidgets.QLabel("X-Axis")
        self.x_axis_cb = QtWidgets.QComboBox()
        for option in feature_options:
            self.x_axis_cb.addItem(option)
        self.x_axis_cb.setCurrentIndex(self.x_axis_cb.findText(defaultXAxis))

        self.x_axis_cb.setToolTip("Choose a feature to plot on the X-Axis")

        x_axis_layout = QtWidgets.QHBoxLayout()
        x_axis_layout.addWidget(x_axis_label)
        x_axis_layout.addWidget(self.x_axis_cb)

        y_axis_label = QtWidgets.QLabel("Y-Axis")
        self.y_axis_cb = QtWidgets.QComboBox()
        self.y_axis_cb.setToolTip("Choose a feature to plot on the Y-Axis")
        for option in feature_options:
            self.y_axis_cb.addItem(option)
        self.y_axis_cb.setCurrentIndex(self.y_axis_cb.findText(defaultYAxis))

        y_axis_layout = QtWidgets.QHBoxLayout()
        y_axis_layout.addWidget(y_axis_label)
        y_axis_layout.addWidget(self.y_axis_cb)

        z_axis_label = QtWidgets.QLabel("Z-Axis")
        self.z_axis_cb = QtWidgets.QComboBox()
        self.z_axis_cb.setToolTip("Choose a feature to plot on the Z-Axis")
        for option in feature_options:
            self.z_axis_cb.addItem(option)
        self.z_axis_cb.setCurrentIndex(self.z_axis_cb.findText(defaultZAxis))

        z_axis_layout = QtWidgets.QHBoxLayout()
        z_axis_layout.addWidget(z_axis_label)
        z_axis_layout.addWidget(self.z_axis_cb)

        axis_layout = QtWidgets.QHBoxLayout()
        axis_layout.addLayout(x_axis_layout)
        axis_layout.addLayout(y_axis_layout)
        axis_layout.addLayout(z_axis_layout)

        tetrode_layout = QtWidgets.QHBoxLayout()
        tetrode_label = QtWidgets.QLabel("Tetrode:")
        self.tetrode_cb = QtWidgets.QComboBox()

        tetrode_layout.addWidget(tetrode_label)
        tetrode_layout.addWidget(self.tetrode_cb)

        spike_parameter_widgets = [tetrode_layout, axis_layout]

        spike_parameter_layout.addStretch(1)
        for i, widget in enumerate(spike_parameter_widgets):
            if 'Layout' in widget.__str__():
                spike_parameter_layout.addLayout(widget)
                spike_parameter_layout.addStretch(1)
            else:
                spike_parameter_layout.addWidget(widget, 0, QtCore.Qt.AlignCenter)
                spike_parameter_layout.addStretch(1)

        self.win = pg.GraphicsWindow()
        self.feature_plot = self.win.addPlot(row=1, col=0, colspan=2,
                                                # viewBox=CustomViewBox(self, self.win)
                                             )

        # --------- Create the Buttons at the bottom of the Main Window ------------- #

        button_layout = QtWidgets.QHBoxLayout()

        self.plot_btn = QtWidgets.QPushButton("Plot")
        self.plot_btn.clicked.connect(lambda: plot_features(self))

        self.quit_btn = QtWidgets.QPushButton("Quit")
        self.quit_btn.clicked.connect(self.close_app)
        self.quit_btn.setShortcut("Ctrl+Q")  # creates shortcut for the quit button
        self.quit_btn.setToolTip('Click to quit gebaSpike!')

        button_order = [self.plot_btn, self.quit_btn]

        for btn in button_order:
            button_layout.addWidget(btn)

        main_window_layout = QtWidgets.QVBoxLayout()

        layout_order = [filename_layout, spike_parameter_layout, self.win, button_layout]

        # ---------------- add all the layouts and widgets to the Main Window's layout ------------ #

        main_window_layout.addStretch(1)  # adds the widgets/layouts according to the order
        for widget in layout_order:
            if 'Layout' in widget.__str__():
                main_window_layout.addLayout(widget)
                main_window_layout.addStretch(1)
            else:
                main_window_layout.addWidget(widget, 0, QtCore.Qt.AlignCenter)
                main_window_layout.addStretch(1)

        main_window_layout.addStretch(1)  # adds stretch to put the version info at the bottom

        self.setLayout(main_window_layout)  # defining the layout of the Main Window

        # center(self)  # centering the window

        self.show()  # shows the window

    def close_app(self):
        """This method will prompt the user, asking if they would like to quit or not"""
        # pop up window that asks if you really want to exit the app ------------------------------------------------

        choice = QtGui.QMessageBox.question(self, "Quitting ",
                                            "Do you really want to exit?",
                                            QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)

        if choice == QtGui.QMessageBox.Yes:
            sys.exit()  # tells the app to quit
        else:
            pass

    def raiseError(self, error):

        if 'TetrodeExistError' in error:

            filename = error.split('!')[1]

            self.choice = QtWidgets.QMessageBox.question(self, "Tetrode Filename Does Not Exist!",
                                                         "The following tetrode filename does not exist: \n%s\nPlease" %
                                                         filename +
                                                         " ensure that the file exists before attempting to plot the"
                                                         " data",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'InvalidSession' in error:

            session = error.split('!')[1]
            self.choice = QtWidgets.QMessageBox.question(self, "Invalid Session Filename!",
                                                         "The following session filename is invalid: \n%s\nPlease" %
                                                         session +
                                                         " ensure that the appropriate files exist for this session.",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'ChooseSession' in error:

            self.choice = QtWidgets.QMessageBox.question(self, "Choose Session Filename!",
                                                         "You have not chosen a session file yet! Please choose a"
                                                         " session filename before proceeding!",
                                                         QtWidgets.QMessageBox.Ok)

    def get_settings(self):
        with open(self.settings_filename, 'r') as f:
            settings = json.load(f)
        return settings

    def overwrite_settings(self):
        # overwrite settings file
        with open(self.settings_filename, 'w') as f:
            json.dump(self.settings, f, sort_keys=True, indent=4)

    def choose_filename(self):
        """
        This method will allow you to choose a filename to analyze.
        :return:
        """

        if 'file_directory' not in self.settings.keys():
            current_filename, filename_filter = QtGui.QFileDialog.getOpenFileName(
                self, caption="Select a '.Set' file!", directory='', filter='Set Files (*.set)')

        else:
            if os.path.exists(self.settings['file_directory']):
                current_filename, filename_filter = QtGui.QFileDialog.getOpenFileName(
                    self, caption="Select a '.Set' file!", directory=self.settings['file_directory'],
                    filter='Set Files (*.set)')
            else:
                current_filename, filename_filter = QtGui.QFileDialog.getOpenFileName(
                    self, caption="Select a '.Set' file!", directory='', filter='Set Files (*.set)')

        # if no file chosen, skip
        if current_filename == '':
            return

        chosen_directory = os.path.dirname(current_filename)

        self.settings['file_directory'] = chosen_directory

        self.overwrite_settings()

        session_valid, error_raised = validate_session(self, current_filename)

        if session_valid:
            # replace the current .set field in the choose .set window with chosen filename
            self.filename.setText(current_filename)
        else:
            if not error_raised:
                self.choice = None
                self.LogError.signal.emit('InvalidSession!%s' % current_filename)
                while self.choice is None:
                    time.sleep(0.1)

    def filename_changed(self):
        """
        This method will run when the filename LineEdit has been changed
        """
        filename = self.filename.text()
        if os.path.exists(filename):
            self.tetrode_cb.clear()

            tetrode_path, tetrode_list = find_tet(self.filename.text())

            for file in tetrode_list:
                tetrode = os.path.splitext(file)[-1][1:]
                self.tetrode_cb.addItem(tetrode)


def launch_gui():
    """
    This function will launch the gebaSpike GUI
    :return:
    """

    app = QtWidgets.QApplication(sys.argv)

    main_window = MainWindow()  # Creating the Main Window
    main_window.raise_()  # raises the Main Window

    sys.exit(app.exec_())  # prevents the window from immediately exiting out


if __name__ == '__main__':
    launch_gui()