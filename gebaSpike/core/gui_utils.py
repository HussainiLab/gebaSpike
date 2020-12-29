
from PyQt5 import QtWidgets, QtCore
from .default_parameters import default_filename
import time
import os

#from .Tint_Matlab import is_tetrode, read_clu
from pyTint.eeg import is_tetrode
from pyTint.utils import read_clu

from collections import Counter

Large_Font = ("Verdana", 12)  # defines two fonts for different purposes (might not be used
Small_Font = ("Verdana", 8)


def center(self):
    """
    centers the window on the screen
    """
    frameGm = self.frameGeometry()
    screen = QtWidgets.QApplication.desktop().screenNumber(QtWidgets.QApplication.desktop().cursor().pos())
    centerPoint = QtWidgets.QApplication.desktop().screenGeometry(screen).center()
    frameGm.moveCenter(centerPoint)
    self.move(frameGm.topLeft())


class Worker(QtCore.QObject):
    """
    This Worker class is for threading using PyQt
    """
    def __init__(self, function, *args, **kwargs):
        super(Worker, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.start.connect(self.run)

    start = QtCore.pyqtSignal(str)

    @QtCore.pyqtSlot()
    def run(self):
        self.function(*self.args, **self.kwargs)


class Communicate(QtCore.QObject):
    """
    A custom pyqtsignal so that errors and popups can be called from the threads
    to the main window
    """
    signal = QtCore.pyqtSignal(str)


def validate_session(self, set_filename):

    # check if the session contains the default filename
    if type(set_filename) == str:
        if set_filename == default_filename:
            # the user has not chosen a filename
            self.choice = None
            self.LogError.signal.emit('ChooseSession')
            while self.choice is None:
                time.sleep(0.1)
            return False, True
    elif type(set_filename) == list:
        for file in set_filename:
            # the user has not chosen a filename
            if file == default_filename:
                # the user has not chosen a filename
                self.choice = None
                self.LogError.signal.emit('ChooseSession')
                while self.choice is None:
                    time.sleep(0.1)
                return False, True

    return True, False


def validate_cut(self, set_filename, cut_filename):

    if not self.multiple_files:
        set_basename = os.path.splitext(set_filename)[0]
        if set_basename in cut_filename:
            return True, False
        return False, False

    # if it is a multiple file session, we will just say that it is valid and later run the validate multiple sessions
    # function and have that override this out come. Could likely just implement the validate multiple functions
    # code here.
    return True, False


def get_spike_count(filename):
    """Reads through the tetrode file and returns the number of spikes"""

    num_spikes = None
    with open(filename, 'rb') as f:
        for line in f:
            if 'data_start' in str(line):
                break
            elif 'num_spikes' in str(line):
                num_spikes = int(line.decode(encoding='UTF-8').split(" ")[1])
    return num_spikes


def get_cut_spike_count(filename):
    """Reads through the cut file and returns the number of spikes in the file"""

    num_spikes = None
    with open(filename, 'rb') as f:
        for line in f:
            if 'data_start' in str(line):
                break
            elif 'Exact_cut_for' in str(line):
                num_spikes = int(line.decode(encoding='UTF-8').split("spikes: ")[1])
    return num_spikes


def validate_multisessions(set_files, cut_filename, tetrode):

    tetrode_spikes = 0

    cut_spikes = 0
    # get the spikes for set files .tetrode file
    for set_file in set_files:
        basename = os.path.splitext(set_file)[0]
        tetrode_filename = basename + '.%s' % tetrode

        if os.path.exists(tetrode_filename):
            tetrode_spikes += get_spike_count(tetrode_filename)

    if os.path.exists(cut_filename):
        if os.path.splitext(cut_filename)[1] == '.cut':
            cut_spikes += get_cut_spike_count(cut_filename)
        elif '.clu' in cut_filename:
            # if it is a clu file it does not have the number of spikes listed so we will read it in
            cut_spikes += len(read_clu(cut_filename))

    return tetrode_spikes == cut_spikes, cut_spikes, tetrode_spikes


def find_tetrodes(self, set_fullpath):
    """finds the tetrode files available for a given .set file if there is a  .cut file existing.

    if multiple set files were provided, then we will find the tetrode values that overlap for the both of them."""

    set_files = []
    if self.multiple_files:
        set_files = set_fullpath.split(', ')
    else:
        set_files = [set_fullpath]

    num_files = len(set_files)

    tetrode_files = {}

    # finds all the tetrode files
    for file in set_files:
        tetrode_path, session = os.path.split(file)
        session, _ = os.path.splitext(session)

        # getting all the files in that directory
        file_list = os.listdir(tetrode_path)

        # acquiring only a list of tetrodes that belong to that set file
        tetrode_list = [os.path.join(tetrode_path, file) for file in file_list
                        if is_tetrode(file, session)]

        # if the .cut or .clu.X file doesn't exist remove from list
        tetrode_list = [file for file in tetrode_list if (
            os.path.exists(
                os.path.join(tetrode_path, '%s_%s.cut' % (
                    os.path.splitext(file)[0], os.path.splitext(file)[1][1:]))) or
            os.path.exists(
                os.path.join(tetrode_path, '%s.clu.%s' % (
                    os.path.splitext(file)[0], os.path.splitext(file)[1][1:])))
        )]

        tetrode_files[file] = tetrode_list

    # count the files to ensure that we have the same amount of tetrode files as we do sessions

    tetrode_count = Counter()
    for tet_files in tetrode_files.values():
        for file in tet_files:
            ext = os.path.splitext(file)[-1]

            if ext in tetrode_count:
                tetrode_count[ext] += 1
            else:
                tetrode_count[ext] = 1

    # we will only include the tetrodes that are existing across all sessions provided
    tetrode_list = []
    for ext in sorted(tetrode_count):
        if tetrode_count[ext] == num_files:
            # this could likely be optimized, but I figure it won't take too long to iterate through these files anyways
            for values in tetrode_files.values():
                ext_files = [file for file in values if os.path.splitext(file)[-1] == ext]
                tetrode_list.extend(ext_files)

    return tetrode_list
