import sys
import numpy as np
from PyQt6.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from numpy import mean, absolute
import math


# calculate the wavelength
def wavelength(frame):
    wave = 0  # cumulative sum
    for val in range(frame.size):
        if val > 0:  # skip first value
            wave += (absolute(frame[val] - frame[val - 1]))
    return wave


# calculate mean absolute value
def calculate_mav(frame):
    return mean(absolute(frame))


# load files from npz
class EMGData:
    def __init__(self):
        phase = "ramp"
        with np.load('data/P09.npz') as fd:  # read data files
            self.signal = fd[f"{phase}_emg"]  # signal data
            self.prompts = fd[f"{phase}_emg_prompts"]  # prompts
            self.trials = fd[f"{phase}_emg_trial"]  # trials
            self.channel = fd["emg_variable_names"]  # channel name


class EMGPlotter(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("EMG Plotter")
        self.showMaximized()

        # figure for emg plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # combo box for prompt and trial selection
        self.prompt_select = QComboBox()
        options = ["1", "2", "3", "4", "5", "6", "7"]
        self.prompt_select.addItems(options)
        self.trial_select = QComboBox()
        options2 = ["1", "2", "3", "4", "5"]
        self.trial_select.addItems(options2)
        self.channel_select = QComboBox()
        options3 = ["1", "2", "3", "4", "5", "6"]
        self.channel_select.addItems(options3)

        # buttons
        update_button = QPushButton("Update")
        update_button.clicked.connect(self.on_button_clicked)
        self.previous_button = QPushButton("Previous Frame")
        self.previous_button.clicked.connect(self.on_previous)
        self.previous_button.setEnabled(False)
        self.next_button = QPushButton("Next Frame")
        self.next_button.clicked.connect(self.on_next)
        self.next_button.setEnabled(False)

        # window layout
        layout = QGridLayout()

        # group input widgets
        self.groupbox = QGroupBox()
        grid1 = QGridLayout()
        # add widgets to box
        grid1.addWidget(QLabel("Trial #"), 0, 0)
        grid1.addWidget(self.trial_select, 0, 1)
        grid1.addWidget(QLabel("Prompt #"), 0, 2)
        grid1.addWidget(self.prompt_select, 0, 3)
        grid1.addWidget(QLabel("Channel #"), 0, 4)
        grid1.addWidget(self.channel_select, 0, 5)
        grid1.addWidget(update_button, 1, 2, 1, 2)
        self.groupbox.setLayout(grid1)
        layout.addWidget(self.groupbox, 0, 0, 1, 1)

        # group features
        self.groupbox = QGroupBox()
        grid2 = QGridLayout()
        # fields to be updated with button press
        self.mav_label = QLabel("")
        self.wave_label = QLabel("")
        # add widgets to box
        grid2.addWidget(QLabel("Features"), 0, 0)
        grid2.addWidget(QLabel("MAV: "), 0, 1)
        grid2.addWidget(self.mav_label, 0, 2)
        grid2.addWidget(QLabel("Wave length: "), 1, 1)
        grid2.addWidget(self.wave_label, 1, 2)
        self.groupbox.setLayout(grid2)
        layout.addWidget(self.groupbox, 0, 1, 1, 2)

        # add plot to gui
        layout.addWidget(self.toolbar, 1, 0, 1, 4)
        layout.addWidget(self.canvas, 2, 0, 1, 4)
        layout.addWidget(self.previous_button, 3, 0, 1, 2)
        layout.addWidget(self.next_button, 3, 2, 1, 2)

        self.switch_button = QPushButton("View Channel Signal")
        self.switch_button.clicked.connect(self.on_switch_button_clicked)
        self.switch_button.setCheckable(True)
        self.switch_button.setEnabled(False)
        # setting default color of button to light-grey
        self.switch_button.setStyleSheet("background-color : lightgrey")
        layout.addWidget(self.switch_button, 0, 3, 1, 1)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def on_switch_button_clicked(self):
        if self.switch_button.isChecked():  # if button is checked
            # setting background color to light-blue
            self.switch_button.setStyleSheet("background-color : lightblue")
            self.dyn_plot()
            self.next_button.setEnabled(False)
            self.previous_button.setEnabled(False)
        else:  # if it is unchecked
            # set background color back to light-grey
            self.switch_button.setStyleSheet("background-color : lightgrey")
            self.on_button_clicked()

    # move to previous or next frames
    def on_previous(self):
        self.x -= 1
        self.frame_plot(self.x)

    def on_next(self):
        self.x += 1
        self.frame_plot(self.x)

    def on_button_clicked(self):
        # get numerical values
        self.prompt = int(self.prompt_select.currentText())
        self.trial = int(self.trial_select.currentText())
        channel = int(self.channel_select.currentText())
        self.generate_data(self.prompt, self.trial, channel)
        self.previous_button.setEnabled(True)
        self.next_button.setEnabled(True)
        self.switch_button.setEnabled(True)

    def generate_data(self, prompt, trial, channel):
        prompt_idx = emg_data.prompts == prompt  # prompt number
        trial_idx = emg_data.trials == trial  # trial number
        subset = emg_data.signal[trial_idx * prompt_idx]  # intersection of trial and prompt
        s = subset[:, channel - 1]  # get data for channel, channel index starts @0
        self.set_data(s, channel)

    def set_data(self, sample, channel_num):
        length = len(sample)  # length in samples
        fs = 2000  # sampling frequency
        dt = 1 / fs  # sampling duration
        T = length * dt  # duration of record
        self.t = np.arange(0, T, dt)  # time scale
        self.time_scale_frame = 0.160  # frame length in sec
        self.len_frames = self.time_scale_frame / dt  # frame length in samples
        self.num_frames = math.floor(length / self.len_frames)  # number of frames
        self.time_scale = np.arange(0, self.time_scale_frame, dt)  # timescale for single frame
        self.channel = channel_num
        self.sample = sample
        self.x = 0
        self.frame_plot(self.x)

    def frame_plot(self, x):
        if x < (self.num_frames * 10):  # if there is still emg data
            # frame bounds
            low = int(x * (self.len_frames / 10))
            upper = int(low + self.len_frames)
            test = self.sample[low:upper]  # extract data for frame
            self.frame = test  # EMG frame

        self.plot_emg(self.time_scale, self.frame)

    def plot_emg(self, time, frame):
        # if signal exists
        if time.size == frame.size:
            self.figure.clear()
            plt = self.figure.add_subplot(111)
            plt.plot(time+(self.x*0.0160), frame)  # plot emg frame
            plt.set_xlabel('Time (s)')
            plt.set_ylabel('Amplitude (V)')
            plt.set_title("Trial " + str(self.trial) + " -  Prompt " + str(self.prompt)
                          + "\n" + emg_data.channel[self.channel - 1] + " -  Frame " + str(self.x + 1))
            # calculate features
            self.mav = calculate_mav(self.frame)
            self.mav_label.setText(str(self.mav) + " volts")
            self.wave = wavelength(self.frame)
            self.wave_label.setText(str(self.wave) + " volts")
            plt.axhline(y=self.mav, color='r', linestyle='dotted', label="MAV")
            plt.text(0+(self.x*0.0160), self.mav, str(self.mav), color='r')
            plt.margins(0)
            plt.legend()
            self.canvas.draw()

    def dyn_plot(self):
        self.figure.clear()
        plt = self.figure.add_subplot(111)
        # plot EMG
        plt.plot(self.t, self.sample)  # plot emg frame
        plt.set_xlabel('Time (s)')
        plt.set_ylabel('Amplitude (V)')
        plt.set_title("Trial " + str(self.trial) + " -  Prompt " + str(self.prompt)
                      + "\n" + emg_data.channel[self.channel - 1] + " -  Frame " + str(self.x + 1))
        # plt.margins(0)
        self.canvas.draw()


# run main program
if __name__ == "__main__":
    emg_data = EMGData()
    app = QApplication(sys.argv)
    window = EMGPlotter()
    window.show()
    sys.exit(app.exec())
