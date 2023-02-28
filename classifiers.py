import sys
import os
import copy
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
import numpy as np
from PyQt6.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QGroupBox, QLabel, QGridLayout, \
    QMainWindow, QWidget, QPushButton, QFileDialog, QMessageBox, QFrame, QAbstractItemView, QLineEdit


def paint_gradient_bar(widget, event):
    painter = QtGui.QPainter(widget)
    # gradient = QtGui.QLinearGradient(0, 0, 0, widget.height())
    gradient = QtGui.QLinearGradient(0, 0, widget.width(), 0)
    gradient.setColorAt(0, QtGui.QColor(254, 0, 2))
    gradient.setColorAt(0.2, QtGui.QColor(216, 0, 39))
    gradient.setColorAt(0.4, QtGui.QColor(161, 1, 93))
    gradient.setColorAt(0.6, QtGui.QColor(99, 0, 158))
    gradient.setColorAt(0.8, QtGui.QColor(68, 0, 204))
    gradient.setColorAt(1, QtGui.QColor(3, 2, 252))
    painter.fillRect(event.rect(), gradient)
    painter.end()


def create_legend():
    legend_widget = QWidget()
    layout = QGridLayout()
    legend_widget.setLayout(layout)
    layout.addWidget(QLabel("Transition Frames"), 0, 0, Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(QLabel("More Concentrated"), 0, 1, Qt.AlignmentFlag.AlignLeft)
    layout.addWidget(QLabel("Less Concentrated"), 0, 3, Qt.AlignmentFlag.AlignRight)
    layout.setRowMinimumHeight(0, 10)
    layout.setRowMinimumHeight(1, 20)
    gradient_bar = QFrame()
    gradient_bar.setFrameShape(QFrame.Shape.Box)
    gradient_bar.setMaximumSize(410, 40)
    gradient_bar.paintEvent = lambda e: paint_gradient_bar(gradient_bar, e)
    layout.addWidget(gradient_bar, 1, 1, 1, 3)
    return legend_widget


def get_colour(value):
    colour = QtGui.QColor(255, 255, 255)
    if value == 0:
        colour = QtGui.QColor(3, 2, 252)
    elif 1 <= value <= 10:
        colour = QtGui.QColor(0, 0, 175)
    elif 10 < value <= 20:
        colour = QtGui.QColor(68, 0, 204)
    elif 20 < value <= 30:
        colour = QtGui.QColor(70, 21, 169)
    elif 30 < value <= 40:
        colour = QtGui.QColor(99, 0, 158)
    elif 40 < value <= 50:
        colour = QtGui.QColor(102, 0, 102)
    elif 50 < value <= 60:
        colour = QtGui.QColor(161, 1, 93)
    elif 60 < value <= 70:
        colour = QtGui.QColor(153, 0, 0)
    elif 70 < value <= 80:
        colour = QtGui.QColor(216, 0, 39)
    elif 80 < value <= 90:
        colour = QtGui.QColor(204, 0, 0)
    elif 90 < value:
        colour = QtGui.QColor(254, 0, 2)
    return colour


# sort dict keys by class
def sort_key(item):
    return item[0][0], item[0][1]


def count_percentage(numbers):
    count = {}
    total = len(numbers)
    for num in numbers:
        if num not in count:
            count[num] = 0
        count[num] += 1
    # if a classifier did not appear during transition, count as 0
    for num in range(1, 8):
        if num not in count:
            count[num] = 0
    for num, num_count in count.items():
        count[num] = round((num_count / total) * 100)
    return count


class Classifier(QMainWindow):
    def __init__(self):
        super().__init__()

        self.file_name = None
        self.steady = None
        self.truth = None
        self.trials = None
        self.prob = None
        self.predictions = None
        self.subset = None
        self.cur_individual = 0
        self.test_data = {}
        self.final_dict = {}
        self.individual = {}
        self.dictio = {1: "NM", 2: "WF", 3: "WE", 4: "WP", 5: "WS", 6: "CG", 7: "HO"}

        # Adding the outermost keys to list (trial number)
        for i in range(1, 9):
            self.test_data[i] = {}

        self.setWindowTitle("EMG Classifier")
        self.showMaximized()
        layout = QGridLayout()

        # source file select
        groupbox0 = QGroupBox()
        grid0 = QGridLayout()
        self.data_set = QLabel("Please Select A Dataset")
        file_button = QPushButton("Select Data")
        file_button.clicked.connect(self.file_select)
        grid0.addWidget(self.data_set, 0, 0, Qt.AlignmentFlag.AlignCenter)
        grid0.addWidget(file_button, 1, 0)
        groupbox0.setLayout(grid0)
        groupbox0.setMaximumSize(200, 100)
        layout.addWidget(groupbox0, 0, 0)

        groupbox1 = QGroupBox()
        grid1 = QGridLayout()
        groupbox1.setLayout(grid1)
        self.data_button = QPushButton("View Individual Data")
        self.data_button.clicked.connect(self.data_button_clicked)
        self.data_button.setCheckable(True)
        self.data_button.setEnabled(False)
        
        self.prev_button = QPushButton("Previous Data")
        self.next_button = QPushButton("Next Data")
        self.prev_button.setFixedWidth(130)
        self.next_button.setFixedWidth(130)
        self.prev_button.clicked.connect(self.on_previous)
        self.next_button.clicked.connect(self.on_next)
        self.prev_button.hide()
        self.next_button.hide()
        self.current_field = QLabel("")
        self.current_field.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_field.hide()
        self.input_box = QLineEdit()  # Create a QLineEdit object
        self.input_box.setFixedWidth(100)  # Set maximum width
        self.input_box.timer = QTimer()  # timer triggers after 3 seconds of inactivity
        self.input_box.timer.setInterval(3000)  # 3000ms = 3s
        self.input_box.timer.setSingleShot(True)
        # Connect the signals and slots
        self.input_box.returnPressed.connect(self.data_update)
        # self.input_box.textChanged.connect(self.on_text_changed)
        # self.input_box.timer.timeout.connect(self.data_update)
        self.input_box.hide()

        grid1.addWidget(self.prev_button, 2, 1)
        grid1.addWidget(self.next_button, 2, 3)
        grid1.addWidget(self.data_button, 0, 1)
        grid1.addWidget(self.current_field, 0, 2)
        grid1.addWidget(self.input_box, 2, 2)
        layout.addWidget(groupbox1, 0, 1)

        groupbox2 = QGroupBox()
        grid2 = QGridLayout()
        groupbox2.setLayout(grid2)
        self.row_button = QPushButton("View All Rows")
        self.row_button.clicked.connect(self.row_button_clicked)
        self.row_button.setCheckable(True)
        grid2.addWidget(self.row_button, 0, 3, 1, 1)
        groupbox2.setMaximumSize(200, 100)
        layout.addWidget(groupbox2, 0, 2)

        groupbox3 = QGroupBox()
        grid3 = QGridLayout()
        grid3.addWidget(create_legend(), 0, 0)
        groupbox3.setLayout(grid3)
        groupbox3.setMaximumSize(700, 100)
        layout.addWidget(groupbox3, 0, 3, 1, 2)

        # output
        groupbox4 = QGroupBox()
        self.grid4 = QGridLayout()
        self.all_tables = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]
        for x in range(len(self.all_tables)):
            cell_size = 45
            height = int(4.3 * cell_size)
            width = int(9.8 * cell_size)
            self.all_tables[x] = self.create_table(width, height, cell_size, x + 1)
            self.all_tables[x].hideRow(x)
            for col in range(self.all_tables[x].columnCount()):
                self.all_tables[x].item(x, col).setBackground(QtGui.QColor(220, 220, 220))

        self.grid4.addWidget(QLabel(), 1, 0)
        self.grid4.addWidget(self.all_tables[0], 0, 0, 1, 1)
        self.grid4.addWidget(self.all_tables[1], 0, 1, 1, 1)
        self.grid4.addWidget(self.all_tables[2], 0, 2, 1, 1)
        self.grid4.addWidget(self.all_tables[3], 1, 0, 1, 1)
        self.grid4.addWidget(self.all_tables[4], 1, 1, 1, 1)
        self.grid4.addWidget(self.all_tables[5], 1, 2, 1, 1)
        self.grid4.addWidget(self.all_tables[6], 2, 1, 1, 1)
        groupbox4.setLayout(self.grid4)
        layout.addWidget(groupbox4, 1, 0, 6, 5)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    # Update data on user input
    def data_update(self):
        data_input = self.input_box.text() + ".npz"
        self.cur_individual = 0
        if self.individual[data_input]:
            self.current_field.setText(str(data_input))
            self.percentage_data(self.individual[data_input])

    # # If the user typed something, start the timer
    # def on_text_changed(self):
    #     self.input_box.timer.start()

    def file_select(self):
        # file_name = QFileDialog.getOpenFileName(self, "", "", "NPZ Files (*.npz)")
        dir_path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if dir_path:
            for file_name in os.listdir(dir_path):
                if os.path.isfile(os.path.join(dir_path, file_name)):
                    self.data_set.setText(os.path.basename(os.path.normpath(dir_path)))
                    self.file_name = file_name
                    file_path = os.path.join(dir_path, file_name)
                    try:
                        with np.load(file_path) as fd:  # read data files
                            self.predictions = fd["predictions"]
                            self.prob = fd["probabilities"]
                            self.trials = fd["trials"]
                            self.truth = fd["ground_truth_table"]
                            self.steady = fd["steady_state_table"]
                            self.individual[file_name] = {}
                            self.data()
                            self.data_button.setEnabled(True)
                    except KeyError:
                        QMessageBox.critical(self, "Error", "Please select a valid file")

    def create_table(self, w, h, b, num):
        table = QTableWidget()
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        h_headers = self.dictio.copy()
        v_headers = h_headers.pop(num)
        header_labels = list(self.dictio.values())
        table.setRowCount(len(header_labels))
        header_labels.append("Frames")
        table.setColumnCount(len(header_labels))
        table.setHorizontalHeaderLabels(header_labels)

        count = 0
        for value in header_labels:
            item = QTableWidgetItem(v_headers + "->" + value)
            table.setVerticalHeaderItem(count, item)
            count += 1

        table.setMaximumSize(w, h)
        for x in range(len(header_labels)):
            table.setColumnWidth(x, b)
            if x == 7:
                table.setColumnWidth(x, b + 20)
            table.setRowHeight(x, round(b / 1.6))
            for y in range(len(header_labels)):
                table.setItem(y, x, QTableWidgetItem(""))
        return table

    def row_button_clicked(self):
        if self.row_button.isChecked():
            self.row_button.setText("View Less Rows")
            for x in range(len(self.all_tables)):
                self.all_tables[x].showRow(x)
                self.all_tables[x].setMaximumSize(441, 220)
        else:  # if it is unchecked
            self.row_button.setText("View All Rows")
            for x in range(len(self.all_tables)):
                self.all_tables[x].hideRow(x)
                self.all_tables[x].setMaximumSize(441, 194)

    def data_button_clicked(self):
        if self.data_button.isChecked():
            self.data_button.setText("View All Data")
            data = list(self.individual.keys())[self.cur_individual]
            self.current_field.setText(str(data))
            self.percentage_data(self.individual[data])
            self.current_field.show()
            self.next_button.show()
            self.prev_button.show()
            self.input_box.show()
        else:  # if it is unchecked
            self.data_button.setText("View Individual Data")
            self.percentage_data(self.final_dict)
            self.cur_individual = 0
            self.current_field.hide()
            self.next_button.hide()
            self.prev_button.hide()
            self.input_box.hide()
            self.input_box.clear()

    def on_previous(self):
        self.cur_individual -= 1
        if self.individual and len(self.individual.keys()) > self.cur_individual >= 0:
            self.input_box.clear()
            data = list(self.individual.keys())[self.cur_individual]
            self.current_field.setText(str(data))
            print(data, self.individual[data])
            self.percentage_data(self.individual[data])

    def on_next(self):
        self.cur_individual += 1
        if self.individual and len(self.individual.keys()) > self.cur_individual >= 0:
            self.input_box.clear()
            data = list(self.individual.keys())[self.cur_individual]
            self.current_field.setText(str(data))
            print(data, self.individual[data])
            self.percentage_data(self.individual[data])

    def data(self):
        for trial_num in range(1, 9):  # all 8 trials **make dynamic**
            trial_idx = self.steady[:, 3] == trial_num  # get class number
            self.subset = self.steady[trial_idx]  # only data for one trial
            classifier = []
            for x in range(len(self.subset)):
                classifier.insert(x, self.subset[x, 2])  # classifier
                cur_frame = self.subset[x, 0]  # current start frame
                if x != 0:
                    prev_frame = self.subset[x - 1, 1]  # previous end frame
                    trial_idx = self.trials == trial_num  # get trial data
                    subset = self.predictions[trial_idx]  # get predictions data
                    if cur_frame - prev_frame > 1:  # if it does not go directly from one class to other
                        self.test_data[trial_num][classifier[x - 1], classifier[x]] = list(
                            np.array(subset[prev_frame:cur_frame]))
            # Sorting the values in the dictionary
            self.test_data[trial_num] = dict(sorted(self.test_data[trial_num].items(), key=sort_key))
        # print(self.file_name, "test", self.test_data)

        #  combine all trial data into one
        data_copy = copy.deepcopy(self.test_data)
        for trial in data_copy:
            inner_dict = data_copy[trial].copy()
            for inner_key in inner_dict:
                if inner_key in self.final_dict:
                    self.final_dict[inner_key].extend(inner_dict[inner_key])
                else:
                    self.final_dict[inner_key] = inner_dict[inner_key]
        self.percentage_data(self.final_dict)

        cur_individual = {}
        #  individual person data
        data_copy2 = copy.deepcopy(self.test_data)
        for trial in data_copy2:
            inner_dict = data_copy2[trial].copy()
            for inner_key in inner_dict:
                if inner_key in cur_individual:
                    cur_individual[inner_key].extend(inner_dict[inner_key])
                else:
                    cur_individual[inner_key] = inner_dict[inner_key]
        self.individual[self.file_name] = cur_individual  # store individual data
        # print(self.file_name, self.individual)

    # def percentage_data(self, trial_num):
    def percentage_data(self, data):
        percentages = {}
        # print("data", data)
        for class_trans, numbers in data.items():
            percentages[class_trans] = dict(sorted(count_percentage(numbers).items()))
        # fill tables
        for class_trans, transition_frames in percentages.items():
            total_frames = len(data[class_trans[0], class_trans[1]])
            for number, percentage in transition_frames.items():
                table_num = class_trans[0] - 1
                col = number - 1
                row = class_trans[1] - 1
                self.all_tables[table_num].setItem(row, col, QTableWidgetItem(str(percentage) + "%"))
                colour = get_colour(percentage)
                self.all_tables[table_num].item(row, col).setBackground(colour)
                self.all_tables[table_num].item(row, col).setForeground(QtGui.QColor('white'))
                self.all_tables[table_num].setItem(row, 7, QTableWidgetItem(str(total_frames)))

        # if all transitions have no probability
        for x in range(0, 6):
            for y in range(0, 6):
                for col in range(self.all_tables[x].columnCount()):
                    item = self.all_tables[x].item(y, col)
                    if item is not None and not item.text() and item.background().color() != QtGui.QColor(220, 220,
                                                                                                          220):
                        self.all_tables[x].setItem(y, col, QTableWidgetItem("-"))
                        self.all_tables[x].setItem(y, 7, QTableWidgetItem("0"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Classifier()
    window.show()
    sys.exit(app.exec())
