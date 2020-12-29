import os
import time
import numpy as np
from .default_parameters import channel_range, max_num_actions
from .gui_utils import validate_session

# from .Tint_Matlab import getspikes, read_cut, read_clu
from pyTint.spike import getspikes
from pyTint.utils import read_cut, read_clu

from .gui_utils import validate_multisessions
from .plot_utils import CustomViewBox, get_channel_color, MultiLine
from .feature_plot import load_features, plot_features
from .waveform_cut_functions import findSpikeSubsample, get_index_from_cell, \
    get_index_from_old_cell, get_cell_from_index, setPlotTitle, get_channel_from_y, validateMoveValue, \
    find_spikes_crossed, get_max_spikes, get_next_action, get_old_index_from_position, get_grid_dimensions, clear_unit
import pyqtgraph as pg
from functools import partial
from PyQt5 import QtGui
from pyqtgraph.Qt import QtCore


def replot_unit(self, index, cell=None):
    """
    This function will re-plot the cell. This is likely called when you have indicated that you wanted to remove the
    spikes from the cell to another value.

    :param self:
    :param index:
    :param cell:
    :return:
    """

    if index is None:
        return

    # get the cell for the color
    if cell is None:
        cell = get_cell_from_index(self, index)

    # skip the 0'th cell, that is the dummy cell in Tint
    if cell == 0:
        return

    setTitle = False
    for channel in np.arange(self.n_channels):
        if index in self.plot_lines.keys():
            self.unit_plots[index][0].removeItem(self.plot_lines[index][channel])
            self.unit_plots[index][0].removeItem(self.avg_plot_lines[index][channel])

        plot_data = self.unit_data[index][channel]

        current_n = plot_data.shape[0]

        plot_data = plot_data[self.cell_subsample_i[cell][channel], :]

        self.plot_lines[index][channel] = MultiLine(
            np.tile(np.arange(self.samples_per_spike), (plot_data.shape[0], 1)),
            plot_data, pen_color=get_channel_color(cell))

        self.avg_plot_lines[index][channel] = MultiLine(np.arange(self.samples_per_spike).reshape((1, -1)),
                                               np.mean(self.unit_data[index][channel], axis=0).reshape((1, -1)),
                                               pen_color='w', pen_width=2)

        self.unit_plots[index][0].addItem(self.plot_lines[index][channel])
        self.unit_plots[index][0].addItem(self.avg_plot_lines[index][channel])

        if not setTitle:
            if cell in self.original_cell_count.keys():
                setPlotTitle(self.unit_plots[index][0], cell, original_cell_count=self.original_cell_count[cell],
                             current_cell_count=current_n)
            else:
                setPlotTitle(self.unit_plots[index][0], cell, current_cell_count=current_n)
            setTitle = True

    # check if there is a popup window for this cell
    if cell in self.PopUpCutWindow.keys():
        # check if the popup window for this cell is active
        if self.PopUpCutWindow[cell].PopUpActive:
            # if self.PopUpCutWindow[cell].cell == cell:
            self.PopUpCutWindow[cell].plot(index, cell)  # re-plot on the popup window


def reconfigure_units(self, unique_cells):
    """
    This function is called when the cell that you want to plot has not been created yet. Thus we might need to
    reconfigure the subplots (add a row or column).

    :param self:
    :param unique_cells:
    :return:
    """

    unique_cells = sorted(unique_cells)

    # number of cells to plot (this includes the newly added cell)
    n_cells = len(unique_cells)

    # the number of rows and columns that we want to have plotted
    rows, cols = get_grid_dimensions(n_cells, method='nper', n=3)

    # initialize the current row and column count
    row = 0
    col = 0

    # we will create a dictionary to contain the old spike plots
    old_plots = {}
    for index, value in self.plot_lines.items():
        old_plots[index] = value
    self.plot_lines = {}

    # we will create a dictionary to contain the old average plots
    old_avgs = {}
    for index, value in self.avg_plot_lines.items():
        old_avgs[index] = value
    self.avg_plot_lines = {}

    # we will create a dictionary to contain the old unit data
    old_unit_data = {}
    for index, value in self.unit_data.items():
        old_unit_data[index] = value
    self.unit_data = {}
    self.unit_data[-1] = old_unit_data[-1]  # preserving the 0'th cell

    # we will create a dictionary to contain the old average plots
    old_unit_plots = {}
    old_plotted_units = []
    for index, value in self.unit_plots.items():
        old_unit_plots[index] = value
        if value[1] != 0:
            old_plotted_units.append(value[1])  # append the cell value
    self.unit_plots = {}
    self.unit_plots[-1] = old_unit_plots[-1]  # preserving the 0'th cell

    # get the old positions of the plots (row, col)
    self.old_positions = {}
    for index, value in self.unit_positions.items():
        self.old_positions[index] = value
    self.unit_positions = {}

    # get the old ROI objects
    old_unit_drag_lines = {}
    for index, value in self.unit_drag_lines.items():
        old_unit_drag_lines[index] = value
        value.hide()
    self.unit_drag_lines = {}

    # get the old viewbox objects
    old_vb = {}
    for index, value in self.vb.items():
        old_vb[index] = value
    self.vb = {}

    max_spike_plots = get_max_spikes(self)
    if self.max_spike_plots is None:
        if max_spike_plots is None:
            return
        else:
            self.max_spike_plots = max_spike_plots

    for index in np.arange(rows*cols):

        if index < len(unique_cells):
            # then we should re-plot these subplot indices

            cell = unique_cells[index]

            # first we need to determine if we need to add a plot or not
            if (row, col) not in self.old_positions.keys():
                '''
                The plot does not exist in this position before, so we will need to add a subplot in this position
                configure the plot appropriately. As well as add another polyline ROI, and define the drag and mouse
                clicks
                '''

                # add the plot
                self.unit_positions[(row, col)] = index
                self.unit_plots[index] = [self.unit_win.addPlot(row=row, col=col,
                                                                viewBox=CustomViewBox(self, self.unit_win)), cell]
                # add the plot
                self.vb[index] = self.unit_plots[index][0].vb

                # configure the plot
                self.unit_plots[index][0].setXRange(0, self.samples_per_spike, padding=0)  # set the x-range
                self.unit_plots[index][0].setYRange(0, -self.n_channels * channel_range, padding=0)
                self.unit_plots[index][0].hideAxis('left')  # remove the y-axis
                self.unit_plots[index][0].hideAxis('bottom')  # remove the x axis
                self.unit_plots[index][0].hideButtons()  # hide the auto-resize button
                self.unit_plots[index][0].setMouseEnabled(x=False, y=False)  # disables the mouse interactions
                self.unit_plots[index][0].enableAutoRange(False, False)

                # add the ROI
                self.unit_drag_lines[index] = pg.PolyLineROI([[0, 0], [30, 30]])
                self.unit_drag_lines[index].hide()
                self.unit_plots[index][0].addItem(self.unit_drag_lines[index])

                # add the drag and mouse click events to the viewbox
                self.vb[index].mouseDragEvent = partial(drag, self, index)  # overriding the drag event
                self.vb[index].mouseClickEvent = partial(mouse_click_event, self, index)

            else:
                '''
                The row/column does exist. we won't need to create a new subplot, but we will need to move
                everything over and delete the old contents.
                '''
                # we will need to find the old index value so we can make sure we can access that plot
                old_plot_index = get_old_index_from_position(self, (row, col))  # get the index

                # defining the new plot use
                self.unit_plots[index] = [old_unit_plots[old_plot_index][0], cell]
                self.unit_positions[(row, col)] = index

                self.vb[index] = self.unit_plots[index][0].vb
                self.vb[index].mouseDragEvent = partial(drag, self, index)  # overriding the drag event
                self.vb[index].mouseClickEvent = partial(mouse_click_event, self, index)

                self.unit_drag_lines[index] = old_unit_drag_lines[old_plot_index]

            if cell in old_plotted_units:
                # figure out if we have plotted this cell before, if so, we can move that data instead of re-plotting

                old_index = get_index_from_old_cell(cell, old_unit_plots)

                current_n = len(self.cell_indices[cell])
                if cell in self.original_cell_count.keys():
                    setPlotTitle(self.unit_plots[index][0], cell,
                                 original_cell_count=self.original_cell_count[cell],
                                 current_cell_count=current_n)
                else:
                    setPlotTitle(self.unit_plots[index][0], cell, current_cell_count=current_n)

                # we are defining the new spike plots and avg plots given the old index
                self.plot_lines[index] = old_plots[old_index]
                self.avg_plot_lines[index] = old_avgs[old_index]

                # moving the line plots
                for channel in self.plot_lines[index].keys():
                    old_unit_plots[old_index][0].removeItem(old_plots[old_index][channel])
                    self.unit_plots[index][0].addItem(self.plot_lines[index][channel])

                # moving the avg plots
                for channel in self.avg_plot_lines[index].keys():
                    old_unit_plots[old_index][0].removeItem(old_avgs[old_index][channel])
                    self.unit_plots[index][0].addItem(self.avg_plot_lines[index][channel])

                # moving the unit data
                self.unit_data[index] = old_unit_data[old_index]

            else:
                # This cell has not been plotted before, so we will need to create the MultiLines

                setTitle = False
                for channel in np.arange(self.n_channels):

                    self.unit_positions[(row, col)] = index

                    # shifting the data so that the next channel resides below the previous
                    # also making the 1st channel start at a y value of 0

                    cell_bool = self.cell_indices[cell]
                    cell_data = self.tetrode_data[:, cell_bool, :]

                    plot_data = cell_data[channel]

                    channel_max = 127
                    plot_data = plot_data - channel * channel_range - channel_max

                    current_n = plot_data.shape[0]
                    if not setTitle:
                        if cell in self.original_cell_count.keys():
                            setPlotTitle(self.unit_plots[index][0], cell,
                                         original_cell_count=self.original_cell_count[cell],
                                         current_cell_count=current_n)
                        else:
                            setPlotTitle(self.unit_plots[index][0], cell, current_cell_count=current_n)
                        setTitle = True

                    cell_data_sub_channel, subsample_i = findSpikeSubsample(plot_data, self.max_spike_plots)

                    if cell not in self.cell_subsample_i.keys():
                        self.cell_subsample_i[cell] = {channel: subsample_i}
                    else:
                        self.cell_subsample_i[cell][channel] = subsample_i

                    plot_data_avg = np.mean(plot_data, axis=0).reshape((1, -1))

                    if index not in self.unit_data.keys():
                        self.unit_data[index] = {channel: plot_data}
                    else:
                        self.unit_data[index][channel] = plot_data

                    if index not in self.plot_lines.keys():

                        self.plot_lines[index] = {
                            channel: MultiLine(
                                np.tile(np.arange(self.samples_per_spike), (cell_data_sub_channel.shape[0], 1)),
                                cell_data_sub_channel, pen_color=get_channel_color(cell))}
                    else:
                        self.plot_lines[index][channel] = MultiLine(
                            np.tile(np.arange(self.samples_per_spike), (cell_data_sub_channel.shape[0], 1)),
                            cell_data_sub_channel, pen_color=get_channel_color(cell))

                    if index not in self.avg_plot_lines.keys():
                        self.avg_plot_lines[index] = {
                            channel: MultiLine(np.arange(self.samples_per_spike).reshape((1, -1)),
                                               plot_data_avg, pen_color='w', pen_width=2)}
                    else:
                        self.avg_plot_lines[index][channel] = MultiLine(
                            np.arange(self.samples_per_spike).reshape((1, -1)),
                            plot_data_avg, pen_color='w', pen_width=2)

                    self.unit_plots[index][0].addItem(self.plot_lines[index][channel])
                    self.unit_plots[index][0].addItem(self.avg_plot_lines[index][channel])
        else:
            # i made this to remove the items from the plots that we don't need anymore, however I end up just
            # removing the subplot itself so this is no longer necessary
            '''
            # then we should check if the plot has any data on it, and remove it since we no longer need it
            if (row, col) in self.old_positions.keys():
                # then we had previously plotted stuff here
                old_plot_index = get_old_index_from_position(self, (row, col))  # get the index

                for channel in old_plots[old_plot_index].keys():
                    old_unit_plots[old_plot_index][0].removeItem(old_plots[old_plot_index][channel])

                # removing the avg plots
                for channel in old_avgs[old_plot_index].keys():
                    old_unit_plots[old_plot_index][0].removeItem(old_avgs[old_plot_index][channel])

            else:
                # there is nothing plotted, so don't worry about it
                pass
            '''
            pass

        col += 1
        if col >= cols:
            col = 0
            row += 1

    # removing any sub_plots that are no longer necessary
    for position in self.old_positions.keys():
        if position not in self.unit_positions.keys():
            old_plot_index = get_old_index_from_position(self, position)  # get the i
            plot_item = old_unit_plots[old_plot_index][0]
            self.unit_win.removeItem(plot_item)

    # removing any old plots that are no longer necessary
    for index, value in old_unit_plots.items():
        cell = value[1]
        if cell not in self.cell_indices.keys():
            for channel in old_plots[index].keys():
                old_unit_plots[index][0].removeItem(old_plots[index][channel])
                old_unit_plots[index][0].removeItem(old_avgs[index][channel])

    self.old_positons = {}
    self.unit_rows = rows
    self.unit_cols = cols


def plot_units(self):
    """
    This function will go and plot all the cells (besides the 0'th as that is the dummy cell).
    It will create the subplots in the self.unit_win plot object. Find the data and indices for each of
    the cells and then plot them.

    :param self:
    :return:
    """
    unique_cells = np.unique(self.cut_data)
    # the 0'th cell is the dummy cell for Tint so we will remove that

    unique_cells = unique_cells[unique_cells != 0]

    cell_bool = np.where(self.cut_data == 0)[0]
    cell_data = self.tetrode_data[:, cell_bool, :]
    self.cell_indices[0] = np.where(self.cut_data == 0)[0]

    for channel, channel_data in enumerate(cell_data):
        self.unit_plots[-1] = [None, 0]
        if -1 not in self.unit_data.keys():
            self.unit_data[-1] = {channel: channel_data}
        else:
            self.unit_data[-1][channel] = channel_data

    n_cells = len(unique_cells)

    rows, cols = get_grid_dimensions(n_cells, method='nper', n=3)

    if self.max_spike_plots is None:
        max_spike_plots = get_max_spikes(self)
        if max_spike_plots is None:
            return
        else:
            self.max_spike_plots = max_spike_plots

    # add the dummy 0'th cell to the cell_indices for the undo functionality to work properly

    row = 0
    col = 0
    for index in np.arange(len(unique_cells)):
        self.unit_positions[(row, col)] = index

        cell = unique_cells[index]

        self.unit_plots[index] = [self.unit_win.addPlot(row=row, col=col,
                                                        viewBox=CustomViewBox(self, self.unit_win)), cell]

        self.vb[index] = self.unit_plots[index][0].vb

        self.unit_plots[index][0].setXRange(0, self.samples_per_spike, padding=0)  # set the x-range
        self.unit_plots[index][0].setYRange(0, -self.n_channels * channel_range, padding=0)
        self.unit_plots[index][0].hideAxis('left')  # remove the y-axis
        self.unit_plots[index][0].hideAxis('bottom')  # remove the x axis
        self.unit_plots[index][0].hideButtons()  # hide the auto-resize button
        self.unit_plots[index][0].setMouseEnabled(x=False, y=False)  # disables the mouse interactions
        self.unit_plots[index][0].enableAutoRange(False, False)

        self.unit_drag_lines[index] = pg.PolyLineROI([[0, 0], [30, 30]])
        self.unit_drag_lines[index].hide()
        self.unit_plots[index][0].addItem(self.unit_drag_lines[index])

        self.vb[index].mouseDragEvent = partial(drag, self, index)  # overriding the drag event
        self.vb[index].mouseClickEvent = partial(mouse_click_event, self, index)

        cell_bool = np.where(self.cut_data == cell)[0]
        cell_data = self.tetrode_data[:, cell_bool, :]

        self.original_cell_count[cell] = cell_data.shape[1]

        if self.samples_per_spike is None:
            self.samples_per_spike = cell_data.shape[2]

        if self.n_channels is None:
            self.n_channels = cell_data.shape[0]

        self.cell_indices[cell] = cell_bool

        # cell_data_sub = findSpikeSubsample(cell_data, self.max_spike_plots)

        title_set = False
        for channel in np.arange(self.n_channels):
            # we will keep the data and the indices

            # shifting the data so that the next channel resides below the previous
            # also making the 1st channel start at a y value of 0
            plot_data = cell_data[channel]
            # channel_max = np.amax(plot_data)
            channel_max = 127
            plot_data = plot_data - channel*channel_range - channel_max
            if not title_set:
                setPlotTitle(self.unit_plots[index][0], cell, original_cell_count=self.original_cell_count[cell],
                             current_cell_count=plot_data.shape[0])
                title_set = True

            cell_data_sub_channel, subsample_i = findSpikeSubsample(plot_data, self.max_spike_plots)

            if cell not in self.cell_subsample_i.keys():
                self.cell_subsample_i[cell] = {channel: subsample_i}
            else:
                self.cell_subsample_i[cell][channel] = subsample_i

            plot_data_avg = np.mean(plot_data, axis=0).reshape((1, -1))

            if index not in self.unit_data.keys():
                self.unit_data[index] = {channel: plot_data}
            else:
                self.unit_data[index][channel] = plot_data

            if index not in self.plot_lines.keys():

                self.plot_lines[index] = {channel: MultiLine(np.tile(np.arange(self.samples_per_spike), (cell_data_sub_channel.shape[0], 1)),
                                                             cell_data_sub_channel, pen_color=get_channel_color(cell))}
            else:
                self.plot_lines[index][channel] = MultiLine(np.tile(np.arange(self.samples_per_spike), (cell_data_sub_channel.shape[0], 1)),
                                                            cell_data_sub_channel, pen_color=get_channel_color(cell))

            if index not in self.avg_plot_lines.keys():
                self.avg_plot_lines[index] = {channel: MultiLine(np.arange(self.samples_per_spike).reshape((1, -1)),
                                                                 plot_data_avg, pen_color='w', pen_width=2)}
            else:
                self.avg_plot_lines[index][channel] = MultiLine(np.arange(self.samples_per_spike).reshape((1, -1)),
                                                                plot_data_avg, pen_color='w', pen_width=2)

            self.unit_plots[index][0].addItem(self.plot_lines[index][channel])
            self.unit_plots[index][0].addItem(self.avg_plot_lines[index][channel])

        col += 1
        if col >= cols:
            col = 0
            row += 1

    self.unit_rows = rows
    self.unit_cols = cols


def plot_session(self):

    self.reset_parameters()

    session_filename = self.filename.text()

    session_valid, error_raised = validate_session(self, session_filename)

    if session_valid:
        tetrode = self.tetrode_cb.currentText()

        if self.multiple_files:
            session_filepaths = self.filename.text().split(', ')
            session_filepaths = [os.path.splitext(file)[0] for file in session_filepaths]

            sessions_valid, cut_spikes, tetrode_spikes = validate_multisessions(session_filepaths,
                                                                                self.cut_filename.text(), tetrode)
            if not sessions_valid:
                self.choice = None
                self.LogError.signal.emit('InvalidMultiSession!%d!%d' % (cut_spikes, tetrode_spikes))
                while self.choice is None:
                    time.sleep(0.1)
                return

        else:
            session_filepaths = [os.path.splitext(self.filename.text())[0]]

        for session_filepath in session_filepaths:
            # session_filepath = os.path.splitext(self.filename.text())[0]
            tetrode_filename = '%s.%s' % (session_filepath, tetrode)

            if os.path.exists(tetrode_filename):

                if self.tetrode_data_loaded is False:

                    ts, ch1, ch2, ch3, ch4, spikeparams = getspikes(tetrode_filename)
                    if self.tetrode_data is None:
                        self.tetrode_data = np.vstack((ch1, ch2, ch3, ch4)).reshape((4, -1, ch1.shape[1]))
                    else:
                        self.tetrode_data = np.hstack((self.tetrode_data, np.vstack((ch1, ch2, ch3, ch4)).reshape((4, -1, ch1.shape[1]))))

                    if self.spike_times is None:
                        self.spike_times = ts
                    else:
                        self.spike_times = np.r_[self.spike_times, ts]

                    self.n_channels = self.tetrode_data.shape[0]
                    self.samples_per_spike = self.tetrode_data.shape[2]

            else:
                # filename does not exist
                self.choice = None
                self.LogError.signal.emit('TetrodeExistError!%s' % tetrode_filename)

                while self.choice is None:
                    time.sleep(0.1)

                self.tetrode_data_loaded = False
                self.tetrode_data = None
                return

        if self.cut_data_loaded is False:
            filename = self.cut_filename.text()  # the data filename (could be .cut or .clu file)
            if os.path.exists(filename):
                if '.clu.' in filename:
                    cut_data = read_clu(filename)
                else:
                    cut_data = read_cut(filename)

                if self.cut_data is None:
                    self.cut_data = cut_data

                self.cut_data_original = self.cut_data.copy()  # keep a copy of the original to revert if we want.
                self.cut_data_loaded = True
            else:
                self.choice = None
                self.LogError.signal.emit('CutExistError!%s' % filename)

                while self.choice is None:
                    time.sleep(0.1)
                return

        self.tetrode_data_loaded = True

        load_features(self)

        plot_features(self)

        plot_units(self)


def get_index_from_roi(self, roi_value):
    for index, value in self.unit_drag_lines.items():
        if value == roi_value:
            return index
    return None


def drag(self, index, ev=None):
    """
    This function will allow us to modify the line ROI's that we created for each Unity graph. This is so
    the users can select which spikes to cut. We have a dictionary containing each of the ROI's (self.unit_drag_lines),
    where the index (the plot index) is the key, and the value is ROI object.

    Each plot has a different corresponding index as the input to drag, so we can make sure to manipulate the correct
    ROI object using the index input.

    self: the main window object containing the attributes that we will be needing
    index: the index value of the plot (order that the plots were created essentially)
    ev: event.
    """

    # global vb, lr
    if ev.button() == QtCore.Qt.LeftButton:
        # the user is using the left
        if not self.drag_active or index != self.last_drag_index:
            for roi in self.active_ROI:
                try:
                    roi.hide()
                except RuntimeError:
                    # likely the subplot was removed
                    pass

            self.unit_drag_lines[index].show()  # showing the LineSegmentROI
            self.active_ROI = [self.unit_drag_lines[index]]
            self.drag_active = True

        # defining the start of the selected region
        points = [[self.vb[index].mapToView(ev.buttonDownPos()).x(),
                  self.vb[index].mapToView(ev.buttonDownPos()).y()],
                  [self.vb[index].mapToView(ev.pos()).x(),
                   self.vb[index].mapToView(ev.pos()).y()]]

        self.unit_drag_lines[index].setPoints(points)
        self.last_drag_index = index
        ev.accept()
    else:
        pg.ViewBox.mouseDragEvent(self.vb[index], ev)


def mouse_click_event(self, index, ev=None):
    """
    This function will override the mouse click event. If the user shift + left mouse clicks on a plot, this will
    launch the PopUpCutWindow (stored as self.PopUpCutWindow), so the user can have a better look at the data.

    If the user middle mouse click it will initiate a cutting of the cell. For this to work the user must have
    had the ROI line intersecting with the data on the plot. It will calculate the slope of the ROI line and
    then determine if this line intersects with any of the data. If the line intersects with the data, it will remove
    the data from this spike, and move it to the self.move_to_channel() attribute from the main Window. It will
    then re-plot both the cell that the spikes were removed from, and the cell tha tthe spikes were moved to.

    :param self: the main window so we can grab any attributes that we need.
    :param index: the index of the plots, essentially the order of which the plots were created
    :param ev: the event that caused this function to run
    :return: None
    """

    # determine if the user used any modifiers (shift, ctrl, etc).
    modifiers = QtGui.QApplication.keyboardModifiers()

    if ev.button() == QtCore.Qt.RightButton:
        # open menu
        pg.ViewBox.mouseClickEvent(self.vb[index], ev)

    elif ev.button() == QtCore.Qt.LeftButton:
        # we want to make sure the user shift + left clicks to initiate the pop up, check that the modifier is a
        # shift modifier.
        if modifiers == QtCore.Qt.ShiftModifier:
            # then you will launch the popup

            cell = get_cell_from_index(self, index)

            if cell not in self.PopUpCutWindow.keys():
                # this cell does not have a popup window, create one
                self.addPopup(cell)
            else:
                # reset the data for this cell
                self.PopUpCutWindow[cell].reset_data()  # clears any old data that might have been on the popup
            self.PopUpCutWindow[cell].plot(index, get_cell_from_index(self, index))  # needs index, cell as inputs
            self.PopUpCutWindow[cell].raise_()

        else:
            # hopefully drag event
            pg.ViewBox.mouseClickEvent(self.vb[index], ev)

    elif ev.button() == QtCore.Qt.MiddleButton:
        # then we will accept the changes

        # perform the cut on the cell
        cut_cell(self, index)


def valid_cut(self, index, isPopup):

    if isPopup:
        # the popup has a channel specific window, so we will check for that as well
        return self.unit_drag_lines in self.active_ROI or self.channel_drag_lines in self.active_ROI
    else:
        # for the main window we will just determine if the index exists in the active_roi
        return self.unit_drag_lines[index] in self.active_ROI


def update_subsample(main, channel_index, data_chan, cell):
    _, subsample_i = findSpikeSubsample(main.unit_data[channel_index][data_chan],
                                        main.max_spike_plots)

    if cell not in main.cell_subsample_i.keys():
        main.cell_subsample_i[cell] = {data_chan: subsample_i}
    else:
        main.cell_subsample_i[cell][data_chan] = subsample_i


def cut_cell(self, index):
    """
    This method will be run whenever the user decides to perform a cut

    :param self:
    :param index:
    :param ev:
    :return:
    """

    # determine if the window is a popup or the main window

    popup = False
    if self.isPopup():
        main = self.mainWindow
        cell = self.cell
        popup = True
    else:
        main = self
        cell = get_cell_from_index(self, index)

    # determine if the cut is valid for the popup or main window (depending on which one it is)
    if valid_cut(self, index, popup):

        # then we have an active ROI
        # we will get the x,y positions (rounded to the nearest int) of the selected line

        try:
            invalid_cell_number = int(self.move_to_channel.text())
        except:
            self.choice = None
            self.LogError.signal.emit('InvalidMoveChannel')
            while self.choice is None:
                time.sleep(0.1)
            return

        if not validateMoveValue(invalid_cell_number):
            self.choice = None
            self.LogError.signal.emit('InvalidMoveChannel')
            while self.choice is None:
                time.sleep(0.1)
            return

        if invalid_cell_number == cell:
            self.choice = None
            self.LogError.signal.emit('SameChannelInvalid')
            while self.choice is None:
                time.sleep(0.1)
            return

        # get the points from the line segment
        channel = None
        points = None
        if popup:
            if self.unit_drag_lines in self.active_ROI:
                points = np.rint(np.asarray(self.unit_drag_lines.getState()['points']))
                # channel that the user drew the line within
                channel = get_channel_from_y(points[0, 1], channel_range=channel_range, n_channels=main.n_channels)
            elif self.channel_drag_lines in self.active_ROI:
                points = np.rint(np.asarray(self.channel_drag_lines.getState()['points']))
                # channel that the user drew the line within
                channel = int(self.channel_number.currentText()) - 1
        else:
            points = np.rint(np.asarray(self.unit_drag_lines[index].getState()['points']))
            # channel that the user drew the line within
            channel = get_channel_from_y(points[0, 1], channel_range=channel_range, n_channels=main.n_channels)

        # determine if the max spike plots value changed
        max_spikes_changed = False
        if main.max_spike_plots is None:
            # main.max_spike_plots will be none if the value has been changed recently
            max_spike_plots = get_max_spikes(main)
            if max_spike_plots is None:
                # the max spike value is not valid, don't execute the cut
                return
            else:
                # the max spike value is valid
                # set the max spike plots value
                main.max_spike_plots = max_spike_plots

                # update the max spikes by re-calculating the subsample index values
                for plotted_cell in main.cell_subsample_i.keys():
                    plotted_channel_index = get_index_from_cell(main, plotted_cell)

                    if any(plotted_cell == cell_value for cell_value in [0]):
                        # skip the invalid channel (done later) and the dummy channel (not plotted)
                        continue

                    for data_chan in np.arange(main.n_channels):
                        update_subsample(main, plotted_channel_index, data_chan, plotted_cell)

                max_spikes_changed = True

        # getting the cell data within the detected channel
        unit_data = main.unit_data[index][channel]
        crossed_cells = find_spikes_crossed(points, unit_data, samples_per_spike=self.samples_per_spike)

        # append the crossed lines to the invalid cell's plot
        invalid_index = get_index_from_cell(main, invalid_cell_number)
        reconfigure = False

        if len(crossed_cells) != 0:
            # remove these spikes from all the channels
            for data_chan in np.arange(main.n_channels):
                # append the crossed lines to the invalid cell's plot
                if invalid_index is not None:
                    # get the invalid data
                    invalid_cell_data = main.unit_data[index][data_chan][crossed_cells, :]
                    # update the invalid_data channel with this current data
                    main.unit_data[invalid_index][data_chan] = np.vstack((main.unit_data[invalid_index][data_chan],
                                                                          invalid_cell_data))

                    # update the plotted subsample as well
                    update_subsample(main, invalid_index, data_chan, invalid_cell_number)
                else:
                    reconfigure = True

                # delete the invalid data from the selected channel
                main.unit_data[index][data_chan] = np.delete(main.unit_data[index][data_chan], crossed_cells, axis=0)

                # recalculate subplot for the channel that the spikes were removed from
                if len(main.unit_data[index][data_chan]) > 0:
                    update_subsample(main, index, data_chan, cell)
                else:
                    # there is no data left, don't need to worry about the sub-sampling anymore
                    if cell in main.cell_subsample_i.keys():
                        main.cell_subsample_i.pop(cell)
                        reconfigure = True

            # check if the cell still exists
            for key, value in main.unit_data[index].items():
                if len(value) == 0:
                    if index != -1:
                        # avoid popping the dummy cell (index -1)
                        main.unit_data.pop(index)
                        reconfigure = True
                        break

            # update the bool
            cell_indices = main.cell_indices[cell]
            # append invalid cells to the new cell number
            invalid_cells = cell_indices[crossed_cells]
            main.cell_indices[cell] = np.delete(cell_indices, crossed_cells)

            # check if there are still indices for this cell, if empty we will remove
            if len(main.cell_indices[cell]) == 0:
                clear_unit(main, cell)  # delete the cell's plots
                if cell in main.original_cell_count.keys():
                    main.original_cell_count.pop(cell)
                if cell != 0:
                    main.cell_indices.pop(cell)
                reconfigure = True

            if invalid_cell_number in main.cell_indices.keys():
                # the cell has existed already within the main window, we can just add to this plot
                main.cell_indices[invalid_cell_number] = np.concatenate((main.cell_indices[invalid_cell_number],
                                                                         invalid_cells))
            else:
                # this cell is not already plotted, have to add the plot and possibly reconfigure
                main.cell_indices[invalid_cell_number] = invalid_cells

                if invalid_cell_number != 0:
                    reconfigure = True

            # add the latest action
            if len(main.latest_actions) == 0 or max_num_actions == 1:
                main.latest_actions = {0: {'action': 'cut', 'fromCell': cell, 'toCell': invalid_cell_number,
                                           'movedCutIndices': invalid_cells}}
            else:
                next_action = get_next_action(main)
                main.latest_actions[next_action] = {'action': 'cut', 'fromCell': cell, 'toCell': invalid_cell_number,
                                                    'movedCutIndices': invalid_cells}

        if popup:
            # re-plot the popup index
            self.plot(index, cell)
            for roi in self.active_ROI:
                roi.hide()

        # re-plot the main Window
        if not reconfigure:
            # update plots for the invalid cell and the cell you removed these spikes from
            # no need to reconfigure
            replot_unit(main, index)
            invalid_index = get_index_from_cell(main, invalid_cell_number)
            if invalid_index != -1:
                replot_unit(main, invalid_index)
        else:
            # we will need to reconfigure the main window possibly, do so
            if cell in main.cell_indices.keys():
                replot_unit(main, index)

            unique_cells = np.asarray(list(main.cell_indices.keys()))
            reconfigure_units(main, list(unique_cells[unique_cells != 0]))

        if max_spikes_changed:
            for plotted_cell in main.cell_indices.keys():
                if plotted_cell == invalid_cell_number or plotted_cell == cell:
                    continue

                plotted_cell_index = get_index_from_cell(main, plotted_cell)
                replot_unit(main, plotted_cell_index)

        if not popup:
            if index in self.unit_drag_lines:
                self.unit_drag_lines[index].hide()

            try:
                self.active_ROI.remove(self.unit_drag_lines[index])
            except ValueError:
                pass
            except KeyError:
                pass

        self.drag_active = False
        main.actions_made = True
