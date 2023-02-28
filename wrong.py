import sys
import os
from PyQt6 import QtGui
from PyQt6.QtCore import Qt
import numpy as np
from PyQt6.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QGroupBox, QComboBox, QLabel, QGridLayout, \
    QMainWindow, QWidget, QPushButton, QFileDialog, QMessageBox, QFrame, QAbstractItemView, QHeaderView, QVBoxLayout


def paint_gradient_bar(widget, event):
    painter = QtGui.QPainter(widget)
    gradient = QtGui.QLinearGradient(0, 0, 0, widget.height())
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
    layout.addWidget(QLabel("Transition Frames"), 0, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(QLabel("More Uncertainty"), 1, 0, alignment=Qt.AlignmentFlag.AlignRight)
    layout.addWidget(QLabel("Less Uncertainty"), 2, 0, alignment=Qt.AlignmentFlag.AlignRight)
    gradient_bar = QFrame()
    gradient_bar.setFrameShape(QFrame.Shape.Box)
    gradient_bar.setMaximumSize(50, 230)
    gradient_bar.paintEvent = lambda e: paint_gradient_bar(gradient_bar, e)
    layout.addWidget(gradient_bar, 1, 1, 2, 2)
    return legend_widget


def get_colour(value):
    colour = QtGui.QColor(255, 255, 255)
    if value == 1:
        colour = QtGui.QColor(3, 2, 252)
    elif 1 < value <= 10:
        colour = QtGui.QColor(68, 0, 204)
    elif 10 < value <= 20:
        colour = QtGui.QColor(99, 0, 158)
    elif 20 < value <= 30:
        colour = QtGui.QColor(161, 1, 93)
    elif 30 < value <= 40:
        colour = QtGui.QColor(216, 0, 39)
    elif 40 < value:
        colour = QtGui.QColor(254, 0, 2)
    return colour


def create_table(w, h, b):

    table = QTableWidget()
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    table.setRowCount(7)
    table.setColumnCount(7)
    headers = ["NM", "WF", "WE", "WP", "WS", "CG", "HO"]
    table.setHorizontalHeaderLabels(headers)
    table.setVerticalHeaderLabels(headers)
    # table.setHorizontalHeader("End State")
    table.setMaximumSize(w, h)
    # gray out intersections
    for x in range(len(headers)):
        table.setColumnWidth(x, b)
        table.setRowHeight(x, b)
        table.setItem(x, x, QTableWidgetItem())
        table.item(x, x).setBackground(QtGui.QColor(220, 220, 220))
    return table


class Classifier(QMainWindow):
    def __init__(self):
        super().__init__()

        self.steady = None
        self.truth = None
        self.trials = None
        self.prob = None
        self.predictions = None
        self.subset = None
        self.test_data = {}
        self.dictio = {1: "NM", 2: "WF", 3: "WE", 4: "WP", 5: "WS", 6: "CG", 7: "HO"}
        # print(self.dictio[1])
        # print((list(self.dictio.keys())[list(self.dictio.values()).index("WF")]))
        # Adding the outermost keys
        for i in range(1, 9):
            self.test_data[i] = {}

        self.setWindowTitle("EMG Classifier")
        self.showMaximized()
        layout = QGridLayout()

        # source file select
        groupbox0 = QGroupBox()
        grid0 = QGridLayout()
        self.file_name = QLabel("Please Select A File")
        file_button = QPushButton("Select File")
        file_button.clicked.connect(self.file_select)
        grid0.addWidget(QLabel("File Name"), 0, 0)
        grid0.addWidget(self.file_name, 0, 1)
        grid0.addWidget(file_button, 0, 2)
        groupbox0.setLayout(grid0)
        layout.addWidget(groupbox0, 0, 0, 1, 2)

        # group input widgets
        groupbox1 = QGroupBox()
        grid1 = QGridLayout()
        self.trial_select = QComboBox()
        options = ["1", "2", "3", "4", "5", "6", "7", "8"]
        self.trial_select.addItems(options)
        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.on_button_clicked)
        self.update_button.setEnabled(False)
        # add widgets to box
        grid1.addWidget(QLabel("Trial #"), 1, 0)
        grid1.addWidget(self.trial_select, 1, 1)
        grid1.addWidget(self.update_button, 1, 2)
        groupbox1.setLayout(grid1)
        layout.addWidget(groupbox1, 0, 2, 1, 2)

        groupboxx = QGroupBox()
        gridx = QGridLayout()
        gridx.addWidget(create_legend(), 0, 0, 1, 1)
        groupboxx.setLayout(gridx)
        layout.addWidget(groupboxx, 1, 0, 3, 1)

        # output
        groupbox2 = QGroupBox()
        self.grid2 = QGridLayout()
        self.table = create_table(555, 550, 75)
        self.table.itemClicked.connect(self.cell_was_clicked)
        table1 = create_table(232, 227, 29)
        table2 = create_table(232, 227, 29)
        table3 = create_table(232, 227, 29)
        table4 = create_table(232, 227, 29)
        table5 = create_table(232, 227, 29)
        table6 = create_table(232, 227, 29)
        table7 = create_table(232, 227, 29)
        table8 = create_table(232, 227, 29)
        self.all_tables = [table1, table2, table3, table4, table5, table6, table7, table8]
        for x in range(9):
            self.all_tables[x - 1].setVisible(0)

        self.end_label = QLabel("End Position")
        self.grid2.addWidget(self.end_label, 0, 1, 1, 3, Qt.AlignmentFlag.AlignCenter)
        self.grid2.addWidget(QLabel("Start Position"), 1, 0, Qt.AlignmentFlag.AlignRight)
        self.grid2.addWidget(QLabel(), 1, 0)
        self.grid2.addWidget(self.table, 1, 1, 1, 4)
        self.grid2.addWidget(table1, 1, 0, 1, 1)
        self.grid2.addWidget(table2, 1, 1, 1, 1)
        self.grid2.addWidget(table3, 1, 2, 1, 1)
        self.grid2.addWidget(table4, 1, 3, 1, 1)
        self.grid2.addWidget(table5, 2, 0, 1, 1)
        self.grid2.addWidget(table6, 2, 1, 1, 1)
        self.grid2.addWidget(table7, 2, 2, 1, 1)
        self.grid2.addWidget(table8, 2, 3, 1, 1)
        groupbox2.setLayout(self.grid2)
        layout.addWidget(groupbox2, 1, 1, 6, 4)
        # group input widgets
        groupbox3 = QGroupBox()
        grid3 = QGridLayout()
        self.trials_button = QPushButton("View All Trials")
        self.trials_button.clicked.connect(self.all_trials_clicked)
        self.trials_button.setEnabled(False)
        # add widgets to box
        grid3.addWidget(self.trials_button, 1, 0)
        groupbox3.setLayout(grid3)
        layout.addWidget(groupbox3, 0, 4, 1, 2)

        groupbox5 = QGroupBox()
        grid5 = QGridLayout()
        grid5.addWidget(QLabel("Transition Predictions"), 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        self.trans_preds = QLabel("")
        grid5.addWidget(self.trans_preds, 1, 0)
        groupbox5.setLayout(grid5)
        layout.addWidget(groupbox5, 1, 5, 3, 1)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def cell_was_clicked(self, item):
        trial = int(self.trial_select.currentText())
        text = item.text()
        print(text)
        row = item.row()
        col = item.column()
        if text == '' or int(text) > 1:
            info = self.test_data[trial][row + 1, col + 1]
            text = '\n'.join(str(x) for x in info)
            str_info = "Predicted classifier while \ntransitioning from " + str(self.dictio[row+1]) \
                       + " (" + str(row + 1) + ") to \n" + str(self.dictio[col+1]) + " (" +\
                       str(col + 1) + "): \n" + '\n'.join(str(x) for x in info)
            self.trans_preds.setText(str(str_info))
        else:
            self.trans_preds.setText("Transitioned from " + str(self.dictio[row+1]) +  " (" + str(row + 1) \
                                     + ") to \n" + str(self.dictio[col+1]) + " (" + str(col + 1)
                                     + ") with 100% confidence")

    def file_select(self):
        file_name = QFileDialog.getOpenFileName(self, "", "", "NPZ Files (*.npz)")
        if file_name:
            file_path = file_name[0]
            file_name = os.path.basename(file_path)
            print(file_path)
            print(file_name)
            self.file_name.setText(file_name)
            try:
                with np.load(file_path) as fd:  # read data files
                    self.predictions = fd["predictions"]
                    self.prob = fd["probabilities"]
                    self.trials = fd["trials"]
                    self.truth = fd["ground_truth_table"]
                    self.steady = fd["steady_state_table"]
                    self.update_button.setEnabled(True)
                    self.trials_button.setEnabled(True)
                    print(self.predictions)
            except KeyError:
                QMessageBox.critical(self, "Error", "Please select a valid file")

    def all_trials_clicked(self):
        self.table.setVisible(0)
        self.grid2.removeWidget(self.end_label)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for x in range(9):
            self.all_tables[x - 1].setVisible(1)
            self.all_tables[x - 1].setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            trial_idx = self.steady[:, 3] == x
            self.subset = self.steady[trial_idx]
            self.data(self.all_tables[x - 1], x)

    def on_button_clicked(self):
        self.grid2.addWidget(self.end_label, 0, 1, 1, 3, Qt.AlignmentFlag.AlignCenter)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setVisible(1)
        for x in range(8):
            self.all_tables[x - 1].setVisible(0)
            self.all_tables[x - 1].setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        # get numerical values
        trial = int(self.trial_select.currentText())
        trial_idx = self.steady[:, 3] == trial
        self.subset = self.steady[trial_idx]
        self.data(self.table, trial)

    def data(self, table, trial_num):
        classifier = []
        transition_frames = []
        for x in range(len(self.subset)):
            classifier.insert(x, self.subset[x, 2])  # classifier
            cur_frame = self.subset[x, 0]  # current start frame
            if x != 0:
                prev_frame = self.subset[x - 1, 1]  # previous end frame
                transition_frames.insert(x, cur_frame - prev_frame)  # transition frames
                # print(classifier[x], classifier[x - 1], str(transition_frames[x - 1]))
                # if cur_frame - prev_frame > 1:
                self.transition_frames(prev_frame, cur_frame, classifier[x - 1], classifier[x], trial_num)
                # grid starts at 0, 0, so we need to subtract 1 to align with classifier
                table.setItem(classifier[x - 1] - 1, classifier[x] - 1,
                              QTableWidgetItem(str(transition_frames[x - 1])))
                colour = get_colour(transition_frames[x - 1])
                table.item(classifier[x - 1] - 1, classifier[x] - 1).setBackground(colour)

    def transition_frames(self, frame1, frame2, prev_class, cur_class, trial_num):
        trial_idx = self.trials == trial_num
        subset = self.predictions[trial_idx]
        self.test_data[trial_num][prev_class, cur_class] = list(np.array(subset[frame1:frame2]))
        print(self.test_data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Classifier()
    window.show()
    sys.exit(app.exec())
