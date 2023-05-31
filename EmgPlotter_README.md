# EMG Plotter README
The EMG Plotter is a Python program that allows users to plot and analyze Electromyography (EMG) signals. 
The program displays an interface that provides users with the ability to select a specific EMG signal from
a pre-loaded npz file, view the plot of the selected signal, and display the calculated features, namely Mean 
Absolute Value (MAV) and Wavelength of the EMG signal.

### Prerequisites
- Python 3.6 or above
- PyQt6
- numpy
- matplotlib
### How to run the program
1. Clone or download the code from the Github repository
2. Open the terminal and navigate to the directory containing the emg-plotter.py file
3. Run the command python emg_plotter.py to launch the program
### How to use the program
Once the program is launched, select the trial, prompt, and channel number from the respective 
dropdown menus and click on the "Update" button. The plot of the selected EMG signal will be displayed 
in the window, along with the calculated MAV and Wavelength features. Users can use the "Previous Frame" 
and "Next Frame" buttons to navigate between different frames of the selected EMG signal.
Users can also use the "View Channel Signal" button to toggle between the raw EMG signal and the filtered EMG signal.