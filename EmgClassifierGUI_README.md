# EMG Classifier
This code provides a graphical user interface (GUI) for an EMG classifier. It allows users to select a dataset and view individual data or all rows of the dataset. The GUI displays tables with color-coded cells representing different classes of EMG signals.

### Requirements
This code requires the following libraries to be installed:
- PyQt6
- numpy
- pandas
- openpyxl

### Running the program
To use the EMG Classifier, follow these steps:
1. Clone or download the code from the Github repository
2. Open the terminal and navigate to the directory containing the emg_classifier_gui.py file
3. Run the command python emg_classifier_gui.py to launch the program

### Using the GUI
Launch the application and select a dataset using the "Select Data" button.

Once a dataset is selected, you can view individual data by clicking the "View Individual Data" button. Use the "Previous Data" and "Next Data" buttons to navigate between individual data entries.

You can also view all rows of the dataset by clicking the "View All Rows" button. This will display color-coded tables representing different classes of EMG signals.

To take a screenshot of the current view, click the "Take Screenshot" button.

Note: This code assumes that the dataset is stored in NPZ files, and the required data arrays (predictions, probabilities, trials, ground_truth_table, and steady_state_table) are present in the files.
