import sys
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from core.gui_utils import validate_session, Communicate, validate_cut, find_tetrodes
from core.default_parameters import project_name, default_filename, defaultXAxis, defaultYAxis, defaultZAxis, openGL, \
    default_move_channel, max_spike_plots, alt_action_button

# from core.Tint_Matlab import find_tetrodes

from core.plot_functions import plot_session, cut_cell, get_index_from_roi
from core.waveform_cut_functions import moveToChannel, maxSpikesChange
from core.undo import undo_function
from core.feature_plot import feature_name_map
from core.PopUpCutting import PopUpCutWindow
from core.writeCut import write_cut, write_clu
import pyqtgraph.opengl as gl
import os
import json
import time
import numpy as np


class MainWindow(QtWidgets.QWidget):

    def __init__(self):
        """
        initializes many of the variables
        """
        super(MainWindow, self).__init__()

        self.setWindowTitle("%s - Main Window" % project_name)  # sets the main window title

        # initializing attributes
        self.plotted_tetrode = None
        self.change_set_with_tetrode = True
        self.multiple_files = False
        self.cut_filename = None
        self.choose_cut_filename_btn = None
        self.feature_win = None
        self.quit_btn = None
        self.filename = None
        self.choose_filename_btn = None
        self.x_axis_cb = None
        self.y_axis_cb = None
        self.z_axis_cb = None
        self.tetrode_cb = None
        self.choice = None  # the current choice for the error popups
        self.plot_btn = None
        self.feature_plot = None

        # initialize list of actions to undo
        self.latest_actions = {}

        # bool for if an action has been made, I suppose we could take the length of the actions attribute
        self.actions_made = False

        # bool representing if the user is dragging the mouse (for drawing the line segments on the graphs)
        self.drag_active = False

        # the graph index that was last dragged upon with the mouse
        self.last_drag_index = None

        self.feature_data = None
        self.tetrode_data = None
        self.tetrode_data_loaded = False
        self.cut_data = None
        self.cut_data_loaded = False
        self.cut_data_original = None
        self.spike_times = None
        self.scatterItem = None
        self.glViewWidget = None
        self.feature_plot_added = False
        self.samples_per_spike = None

        # keep a list of the positions that the cells are plotted in
        self.unit_positions = {}

        # not all the spikes are plotted at once, so we will keep a dict of which subsample is plotted
        self.cell_subsample_i = {}

        self.unit_drag_lines = {}
        self.active_ROI = []
        self.unit_data = {}

        self.xline = None
        self.yline = None
        self.zline = None

        self.max_spike_plots = None

        self.n_channels = None

        self.invalid_channel = None

        self.cell_indices = {}

        self.spike_colors = None

        self.original_cell_count = {}

        self.unit_plots = {}
        self.vb = {}
        self.plot_lines = {}
        self.avg_plot_lines = {}
        self.unit_rows = 0
        self.unit_cols = 0

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

        self.IMG_DIR = os.path.join(self.PROJECT_DIR, 'img')  # settings directory

        # create Settings Directory
        if not os.path.exists(self.SETTINGS_DIR):
            os.mkdir(self.SETTINGS_DIR)

        # create settings file
        if not os.path.exists(self.settings_filename):
            with open(self.settings_filename, 'w') as f:
                json.dump({}, f)

        # setting PyQtGraph settings
        # pg.setConfigOption('background', 'w')
        # pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(antialias=True)

        self.PopUpCutWindow = {}

        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('GTK+'))

        self.setWindowIcon(QtGui.QIcon(os.path.join(self.IMG_DIR, 'GEBA_Logo.png')))  # declaring the icon image
        self.initialize()  # initializes the main window

    def keyPressEvent(self, event):
        """
        This method will occur when the main window is on top and the user presses a button

        :param event:
        :return:
        """
        if type(event) == QtGui.QKeyEvent:
            # here accept the event and do something
            if event.key() == alt_action_button:
                # check if there is a popup that is shown
                if len(self.PopUpCutWindow) > 0:
                    pass

                if len(self.active_ROI) == 1:
                    # get the index
                    index = get_index_from_roi(self, self.active_ROI[0])
                    cut_cell(self, index)
                elif len(self.active_ROI) > 1:
                    # there shouldn't be more than 1 ROI open
                    self.active_ROI = self.active_ROI[-1]
                    index = get_index_from_roi(self, self.active_ROI[0])
                    # this shouldn't happen, lets remove all the ROI's
                    cut_cell(self, index)

            event.accept()
        else:
            event.ignore()

    def isPopup(self):
        """
        This function is is for the cutting functionality as it will tell the cut function that it is not a popup
        window, and thus to treat it as such.
        """
        return False

    def initialize(self):
        """
        This method will initialize the Main Window, create all the widgets and what not
        :return:
        """

        # ---------------- session filename options ------------------------------------------

        filename_grid_layout = QtWidgets.QGridLayout()

        filename_label = QtWidgets.QLabel("Filename:")
        self.filename = QtWidgets.QLineEdit()
        self.filename.setText(default_filename)
        self.filename.textChanged.connect(self.filename_changed)
        self.filename.setToolTip('The name of the session that you will analyze in gebaSpike!')
        self.choose_filename_btn = QtWidgets.QPushButton("Choose Filename")
        self.choose_filename_btn.setToolTip('This button will allow you to choose the filename to analyze!')
        self.choose_filename_btn.clicked.connect(self.choose_filename)

        filename_grid_layout.addWidget(self.choose_filename_btn, *(0, 0))
        filename_grid_layout.addWidget(filename_label, *(0, 1), QtCore.Qt.AlignRight)

        # ------------------- cut filename options ---------------------------------

        cut_filename_label = QtWidgets.QLabel("Cut Filename:")
        self.cut_filename = QtWidgets.QLineEdit()
        self.cut_filename.setText(default_filename)
        self.cut_filename.setToolTip('The name of the cut file containing the sorted values!')
        self.choose_cut_filename_btn = QtWidgets.QPushButton("Choose Cut Filename")
        self.choose_cut_filename_btn.setToolTip('This button will allow you to choose a cut file to use!')
        self.choose_cut_filename_btn.clicked.connect(self.choose_cut_filename)

        line_edit_layout = QtWidgets.QVBoxLayout()
        line_edit_layout.addWidget(self.filename)
        line_edit_layout.addWidget(self.cut_filename)

        filename_grid_layout.addWidget(self.choose_cut_filename_btn, *(1, 0))
        filename_grid_layout.addWidget(cut_filename_label, *(1, 1), QtCore.Qt.AlignRight)

        filename_grid_layout.addLayout(line_edit_layout, *(0, 2), 2, 1)

        # -----------

        max_spike_label = QtWidgets.QLabel("Max Plot Spikes:")
        self.max_spike_plots_text = QtWidgets.QLineEdit()
        self.max_spike_plots_text.setToolTip("This is the maximum number of spikes to plot.")
        self.max_spike_plots_text.setText(str(max_spike_plots))
        self.max_spike_plots_text.textChanged.connect(lambda: maxSpikesChange(self, 'main'))

        max_spikes_layout = QtWidgets.QHBoxLayout()
        max_spikes_layout.addWidget(max_spike_label)
        max_spikes_layout.addWidget(self.max_spike_plots_text)

        # ------- move invalid -----------

        move_to_channel_label = QtWidgets.QLabel("Move to Cell:")
        self.move_to_channel = QtWidgets.QLineEdit()
        self.move_to_channel.setToolTip("This is the channel that you will move selected cells too.")
        self.move_to_channel.setText(default_move_channel)
        self.move_to_channel.textChanged.connect(lambda: moveToChannel(self, 'main'))

        move_to_channel_layout = QtWidgets.QHBoxLayout()
        move_to_channel_layout.addWidget(move_to_channel_label)
        move_to_channel_layout.addWidget(self.move_to_channel)

        # -------- spike parameter options -------------------------------------

        feature_options = ["None"] + list(feature_name_map.keys())

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
        self.tetrode_cb.currentIndexChanged.connect(self.tetrode_changed)
        tetrode_layout.addWidget(tetrode_label)
        tetrode_layout.addWidget(self.tetrode_cb)

        spike_parameter_widgets = [tetrode_layout, axis_layout, move_to_channel_layout, max_spikes_layout]

        spike_parameter_layout.addStretch(1)
        for i, widget in enumerate(spike_parameter_widgets):
            if 'Layout' in widget.__str__():
                spike_parameter_layout.addLayout(widget)
                spike_parameter_layout.addStretch(1)
            else:
                spike_parameter_layout.addWidget(widget, 0, QtCore.Qt.AlignCenter)
                spike_parameter_layout.addStretch(1)

        if openGL:
            self.feature_win = pg.GraphicsWindow()
            feature_win_layout = QtWidgets.QGridLayout()
            self.feature_win.setLayout(feature_win_layout)
            self.glViewWidget = gl.GLViewWidget()
            self.glViewWidget.setBackgroundColor('k')

            feature_win_layout.addWidget(self.glViewWidget)
        else:
            self.feature_win = MatplotlibWidget()
            self.feature_win.toolbar.hide()  # hide the toolbar

        self.unit_win = pg.GraphicsWindow()

        plot_layout = QtWidgets.QHBoxLayout()
        for _object in [self.feature_win, self.unit_win]:
            plot_layout.addWidget(_object)

        # ------------------------------------ version information -------------------------------------------------

        vers_label = QtWidgets.QLabel("gebaSpike V1.0.10")

        # --------- Create the Buttons at the bottom of the Main Window ------------- #

        button_layout = QtWidgets.QHBoxLayout()

        self.plot_btn = QtWidgets.QPushButton("Plot")
        self.plot_btn.clicked.connect(self.plotFunc)

        self.save_btn = QtWidgets.QPushButton("Save Cut")
        self.save_btn.clicked.connect(self.save_function)
        self.save_btn.setToolTip('Save to the current cut file')

        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.undo_btn.clicked.connect(lambda: undo_function(self))
        self.undo_btn.setToolTip('Click to undo previous action!')

        self.quit_btn = QtWidgets.QPushButton("Quit")
        self.quit_btn.clicked.connect(self.close_app)
        self.quit_btn.setShortcut("Ctrl+Q")  # creates shortcut for the quit button
        self.quit_btn.setToolTip('Click to quit gebaSpike!')

        button_order = [self.plot_btn,
                        self.save_btn,
                        self.undo_btn,
                        self.quit_btn]

        for btn in button_order:
            button_layout.addWidget(btn)

        main_window_layout = QtWidgets.QVBoxLayout()

        layout_order = [filename_grid_layout, spike_parameter_layout, plot_layout, button_layout, vers_label]
        # do you want to add stretch to the widgets/layouts within the layout order?
        add_Stretch = [False, False, False, False, False]
        # do you want to center the widgets/layouts within the layout order?
        align_center = [True, True, True, True, False]

        # ---------------- add all the layouts and widgets to the Main Window's layout ------------ #

        # main_window_layout.addStretch(1)  # adds the widgets/layouts according to the order
        for widget, addStretch, alignCenter in zip(layout_order, add_Stretch, align_center):
            if 'Layout' in widget.__str__():
                main_window_layout.addLayout(widget)
                if addStretch:
                    main_window_layout.addStretch(1)
            else:
                if alignCenter:
                    main_window_layout.addWidget(widget, 0, QtCore.Qt.AlignCenter)
                else:
                    main_window_layout.addWidget(widget, 0)
                if addStretch:
                    main_window_layout.addStretch(1)

        self.setLayout(main_window_layout)  # defining the layout of the Main Window

        self.show()  # shows the window

    def addPopup(self, cell):
        """
        This method will modify the self.PopUpCutWindow, so that we keep track of which popup window belongs to which
        cell

        :param cell:
        :return:
        """
        self.PopUpCutWindow[cell] = PopUpCutWindow(self)

    def save_function(self):
        """
        this method will save the .cut file

        :return:
        """
        if self.cut_filename.text() == default_filename:
            return

        save_filename = os.path.realpath(self.cut_filename.text())

        if os.path.exists(save_filename):
            self.choice = None
            self.LogError.signal.emit('OverwriteCut!%s' % save_filename)
            while self.choice is None:
                time.sleep(0.1)

            if self.choice != QtWidgets.QMessageBox.Yes:
                return

        if len(self.tetrode_data) == 0:
            return

        # organize the cut data
        n_spikes_expected = self.tetrode_data.shape[1]
        n_spikes = len(np.asarray([item for sublist in self.cell_indices.values() for item in sublist]))

        # check that with the manipulation of the spikes, that we still have the correct number of spikes
        if n_spikes != n_spikes_expected:
            self.choice = None
            self.LogError.signal.emit('cutSizeError')
            while self.choice is None:
                time.sleep(0.1)
            return

        # we will check if we are missing some of the spikes somehow. If we kept track of them, then the indices from
        # the spikes, when sorted, should produce an array from 0 -> N-1 spikes.
        if not np.array_equal(np.sort(np.asarray([item for sublist in self.cell_indices.values() for item in sublist])),
                          np.arange(len(self.cut_data_original))):
            self.choice = None
            self.LogError.signal.emit('cutIndexError')
            while self.choice is None:
                time.sleep(0.1)
            return

        cut_values = np.zeros(n_spikes)
        for cell, cell_indices in self.cell_indices.items():
            cut_values[cell_indices] = cell

        if '.clu.' in save_filename:
            # save the .clu filename
            write_clu(save_filename, cut_values)
            self.choice = None
            self.LogError.signal.emit('saveCompleteClu')
            while self.choice is None:
                time.sleep(0.1)
            self.actions_made = False

        else:
            # save the cut filename
            write_cut(save_filename, cut_values)
            self.choice = None
            self.LogError.signal.emit('saveComplete')
            while self.choice is None:
                time.sleep(0.1)
            self.actions_made = False

    def close_app(self):
        """This method will prompt the user, asking if they would like to quit or not"""
        # pop up window that asks if you really want to exit the app ------------------------------------------------

        choice = QtWidgets.QMessageBox.question(self, "Quitting ",
                                                "Do you really want to exit?",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        if choice == QtWidgets.QMessageBox.Yes:
            sys.exit()  # tells the app to quit
        else:
            pass

    def raiseError(self, error):
        """
        This method will be called whenever an error has occurred, and will generally raise Message Box that the user
        will interact with.

        :param error:
        :return:
        """
        if 'TetrodeExistError' in error:
            filename = error.split('!')[1]
            self.choice = QtWidgets.QMessageBox.question(self, "Tetrode Filename Does Not Exist!",
                                                         "The following tetrode filename does not exist: \n%s\nPlease" %
                                                         filename +
                                                         " ensure that the file exists before attempting to plot the"
                                                         " data!",
                                                         QtWidgets.QMessageBox.Ok)

        if 'CutExistError' in error:
            filename = error.split('!')[1]
            self.choice = QtWidgets.QMessageBox.question(self, "Cut/Clu File Exist ERror!",
                                                         "The following .cut/.clu filename does not exist: \n%s\nPlease" %
                                                         filename +
                                                         " ensure that the file exists before attempting to plot the"
                                                         " data!",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'InvalidSession' in error:
            session = error.split('!')[1]
            self.choice = QtWidgets.QMessageBox.question(self, "Invalid Session Filename!",
                                                         "The following session filename is invalid: \n%s\nPlease" %
                                                         session +
                                                         " ensure that the appropriate files exist for this session.",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'InvalidMultiSession' in error:
            cut_spikes = error.split('!')[1]
            tetrode_spikes = error.split('!')[2]
            self.choice = QtWidgets.QMessageBox.question(self, "Invalid Multi Sessions!",
                                                         "You are choosing to combine multiple session files! The "
                                                         "chosen .cut filename (with the current tetrode) has " + str(
                                                         cut_spikes) + " spikes"
                                                         " whereas the combined tetrode spikes for the chosen session "
                                                         "files add up to " + tetrode_spikes + " spikes! Make sure you "
                                                         "choose the correct .cut file for the chosen .set files!",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'cutIndexError' in error:
            self.choice = QtWidgets.QMessageBox.question(self, "Cut Index Error!",
                                                         "The cut output is missing some of the spikes!",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'cutSizeError' in error:
            self.choice = QtWidgets.QMessageBox.question(self, "Cut Size Error!",
                                                         "Trying to save an inappropriate number of spikes, cannot " +
                                                         "save cut file!",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'InvalidCut' in error:
            session = error.split('!')[1]
            self.choice = QtWidgets.QMessageBox.question(self, "Invalid Cut/Clu Filename!",
                                                         "The following Cut filename is invalid: \n%s\nPlease" %
                                                         session +
                                                         " ensure that this .cut/clu filename belongs to this session!",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'ChooseSession' in error:
            self.choice = QtWidgets.QMessageBox.question(self, "Choose Session Filename!",
                                                         "You have not chosen a session file yet! Please choose a"
                                                         " session filename before proceeding!",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'InvalidMoveChannel' in error:
            self.choice = QtWidgets.QMessageBox.question(self, "Invalid Move to Cell Value!",
                                                         "The value you have chosen for the 'Move to Cell' value is "
                                                         "invalid, please choose a valid value before continuing!",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'SameChannelInvalid' in error:
            self.choice = QtWidgets.QMessageBox.question(self, "Same Channel Error!",
                                                         "The value you have chosen for the 'Move to Cell' value is "
                                                         "the same as the cell you are cutting from! If you would like "
                                                         "to move these selected spikes to a different channel, please "
                                                         "choose another channel!",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'ActionsMade' in error:
            self.choice = QtWidgets.QMessageBox.question(self, "Are you sure?",
                                                         "Any unsaved actions will be lost when you"
                                                         " reload this cut file. Are you sure you want to continue?",
                                                         QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        elif 'OverwriteCut' in error:
            cut_file = os.path.realpath(error.split('!')[1])
            self.choice = QtWidgets.QMessageBox.question(self, "Cut Filename Exists",
                                                         "The following cut filename exists:\n\n%s\n\n Are you sure you" 
                                                         " want to overwrite this file?" % cut_file,
                                                         QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        elif 'saveComplete' in error:
            self.choice = QtWidgets.QMessageBox.question(self, "Save Complete",
                                                         "Your '.cut' file has been saved successfully!",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'saveCompleteClu' in error:
            self.choice = QtWidgets.QMessageBox.question(self, "Save Complete",
                                                         "Your '.clu' file has been saved successfully!",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'invalidMaxSpikes' in error:
            self.choice = QtWidgets.QMessageBox.question(self, "Invalid Max Spikes",
                                                         "The number chosen for Max Spikes is invalid!",
                                                         QtWidgets.QMessageBox.Ok)

    def get_settings(self):
        """
        This methoid will read the settings filename and return any settings

        :return:
        """
        settings = {}
        if os.path.exists(self.settings_filename):
            with open(self.settings_filename, 'r') as f:
               settings = json.load(f)
        return settings

    def overwrite_settings(self):
        """
        This method will overwrite the settings file with the current settings

        :return:
        """
        # overwrite settings file
        with open(self.settings_filename, 'w') as f:
            json.dump(self.settings, f, sort_keys=True, indent=4)

    def reset_parameters(self):
        """
        This method will reset the parameters, generally used when switching sessions so we can start fresh.
        :return:
        """
        self.plotted_tetrode = None
        self.feature_data = None
        self.tetrode_data = None
        self.tetrode_data_loaded = False
        self.cut_data = None
        self.cut_data_loaded = False
        self.cut_data_original = None
        self.n_channels = None
        self.spike_times = None

        self.max_spike_plots = None

        self.cell_indices = {}
        self.plot_lines = {}
        self.avg_plot_lines = {}
        self.vb = {}
        self.active_ROI = []
        self.unit_data = {}
        self.cell_subsample_i = {}
        self.unit_positions = {}
        self.original_cell_count = {}

        for popup in self.PopUpCutWindow.values():
            popup.close()

        self.PopUpCutWindow = {}

        self.latest_actions = {}

        self.actions_made = False

        self.last_drag_index = None

        self.drag_active = False

        self.unit_drag_lines = {}

        self.samples_per_spike = None
        self.spike_colors = None

        self.unit_plots = {}
        self.unit_rows = 0
        self.unit_cols = 0

        self.reset_plots()

    def reset_plots(self):
        """
        This method will clear the plots

        :return:
        """
        if hasattr(self, 'scatterItem'):
            if self.scatterItem is not None:
                self.glViewWidget.removeItem(self.scatterItem)
                self.scatterItem = None
        self.unit_win.clear()

    def choose_cut_filename(self):
        if 'file_directory' not in self.settings.keys():
            current_filename, filename_filter = QtWidgets.QFileDialog.getOpenFileName(
                self, caption="Select a '.Set' file!", directory='', filter='Set Files (*.set)')

        else:
            if os.path.exists(self.settings['file_directory']):
                current_filename, filename_filter = QtWidgets.QFileDialog.getOpenFileName(
                    self, caption="Select a '.Cut/.Clu' file!", directory=self.settings['file_directory'],
                    filter='Cut Files (*.cut *.clu*)')
            else:
                current_filename, filename_filter = QtWidgets.QFileDialog.getOpenFileName(
                    self, caption="Select a '.Cut/.Clu' file!", directory='', filter='Cut Files (*.cut *.clu*)')

        # if no file chosen, skip
        if current_filename == '':
            return

        chosen_directory = os.path.dirname(current_filename)

        self.settings['file_directory'] = chosen_directory

        self.overwrite_settings()

        self.reset_parameters()

        self.cut_filename.setText(os.path.realpath(current_filename))

    def plotFunc(self):
        """
        This method will be called when the Plot button is pressed.

        :return:
        """

        # validate the cut filename
        cut_valid, error_raised = validate_cut(self, self.filename.text(), self.cut_filename.text())

        if not cut_valid:
            self.choice = None
            self.LogError.signal.emit('InvalidCut!%s' % self.cut_filename.text())
            while self.choice is None:
                time.sleep(0.1)
            return

        # check if there were any actions made, this is because the attributes will be reset
        if self.actions_made is True:
            # there were actions made, raise the error so the user can decide to continue or not

            # we don't want this to pop-up whenever switch tetrodes, so check if the plotted data is from the
            # current chosen tetrode
            if self.tetrode_cb.currentText() == self.plotted_tetrode:
                self.choice = None
                self.LogError.signal.emit('ActionsMade')
                while self.choice is None:
                    time.sleep(0.1)

                # check which option the user chose
                if self.choice == QtWidgets.QMessageBox.Yes:
                    # user chose the re-plot anyways
                    self.reset_plots()
                    self.reset_parameters()
                    plot_session(self)
            else:
                # the tetrodes are different, just plot the session
                self.reset_plots()
                self.reset_parameters()
                plot_session(self)
        else:
            # there were no actions made, simply plot the session
            plot_session(self)

        self.plotted_tetrode = self.tetrode_cb.currentText()

    def choose_filename(self):
        """
        This method will allow you to choose a .set filename to analyze.
        :return:
        """

        if 'file_directory' not in self.settings.keys():
            current_filename, filename_filter = QtWidgets.QFileDialog.getOpenFileNames(
                self, caption="Select a '.Set' file!", directory='', filter='Set Files (*.set)')

        else:
            if os.path.exists(self.settings['file_directory']):
                current_filename, filename_filter = QtWidgets.QFileDialog.getOpenFileNames(
                    self, caption="Select a '.Set' file!", directory=self.settings['file_directory'],
                    filter='Set Files (*.set)')
            else:
                current_filename, filename_filter = QtWidgets.QFileDialog.getOpenFileNames(
                    self, caption="Select a '.Set' file!", directory='', filter='Set Files (*.set)')

        # if no file chosen, skip
        if len(current_filename) == 0:
            return

        if len(current_filename) == 1:
            # then the user has selected only one
            current_filename = current_filename[0]

            # check if the session is valid
            session_valid = validate_session(self, current_filename)

            if session_valid:
                self.multiple_files = False

                # updated the chosen directory settings value so we remember our latest chosen directory
                chosen_directory = os.path.dirname(current_filename)
                self.settings['file_directory'] = chosen_directory
                self.overwrite_settings()

                # replace the current .set field in the choose .set field with chosen filename
                self.filename.setText(os.path.realpath(current_filename))

                self.reset_parameters()

            else:
                # the session is not valid, raise the error message
                self.choice = None
                self.LogError.signal.emit('InvalidSession!%s' % current_filename)
                while self.choice is None:
                    time.sleep(0.1)
        elif len(current_filename) > 1:
            # then we have selected multiple files

            # check if the session is valid
            session_valid = validate_session(self, current_filename)

            if session_valid:
                self.multiple_files = True

                # updated the chosen directory settings value so we remember our latest chosen directory
                chosen_directory = os.path.dirname(current_filename[0])
                self.settings['file_directory'] = chosen_directory
                self.overwrite_settings()

                # replace the current .set field in the choose .set field with chosen filename
                self.filename.setText(os.path.realpath(', '.join(current_filename)))

                self.reset_parameters()

            else:
                # the session is not valid, raise the error message
                self.choice = None
                self.LogError.signal.emit('InvalidSession!%s' % ', '.join(current_filename))
                while self.choice is None:
                    time.sleep(0.1)
        else:
            return

    def set_cut_filename(self):
        """
        When you choose a session filename, it will trigger this function which will automatically set the .cut
        filename for the user.

        :return:
        """

        if not self.multiple_files:
            filename = self.filename.text()

            try:
                tetrode = int(self.tetrode_cb.currentText())
            except ValueError:
                return

            cut_filename = '%s_%d.cut' % (
                os.path.splitext(filename)[0], tetrode)
            clu_filename = '%s.clu.%d' % (
                os.path.splitext(filename)[0], tetrode)
            if os.path.exists(cut_filename):
                self.cut_filename.setText(os.path.realpath(cut_filename))
            elif os.path.exists(clu_filename):
                self.cut_filename.setText(os.path.realpath(clu_filename))

    def tetrode_changed(self):
        """
        Upon changing of the tetrode drop-menu, the .cut file will also need to be changed (trigger that change)

        :return:
        """
        # we will update the cut_filename
        if self.change_set_with_tetrode:
            self.set_cut_filename()
            # self.reset_parameters()

    def filename_changed(self):
        """
        This method will run when the filename LineEdit has been changed.

        It will essentially find the active tetrodes and populate the drop-down menu.
        """

        filename = self.filename.text()

        if self.multiple_files:
            filename = filename.split(', ')

        else:
            filename = [filename]

        # ensure that the files exist
        for file in filename:
            if not os.path.exists(file):
                return

        tetrodes = []

        self.tetrode_cb.clear()

        tetrode_list = find_tetrodes(self, self.filename.text())

        # get the extension value (excluding the .) so we can create a list of tetrode integers

        for file in tetrode_list:
            tetrode = os.path.splitext(file)[-1][1:]
            tetrodes.append(tetrode)

        # make a list of added tetrodes
        added_tetrodes = []
        for tetrode in sorted(tetrodes):
            # check if the tetrode value has been added already
            if tetrode in added_tetrodes:
                # continue if already added
                continue

            # add the item to the list containing the tetrode value
            self.tetrode_cb.addItem(tetrode)

            # add the tetrode value to the added_tetrodes list
            added_tetrodes.append(tetrode)

        # set the cut_filename
        self.set_cut_filename()


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
