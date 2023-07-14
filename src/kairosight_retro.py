#Kairosight 3.0 - Open Source Software Update

#This is an open-source code repository for Kairosight 3.0, an updated version of the 
#Kairosight software.  this Python code provides advanced signal analysis and visualization 
#tools for research and analysis purposes.

#Kairosight 3.0 is released under an open license, allowing users to modify and utilize 
#the codebase while adhering to proper citation practices. We kindly request that any usage 
#or modifications of this code be appropriately cited to acknowledge the contributions of 
#the original developers.

#The new features introduced in Kairosight 3.0 include:

# 1.User-defined masking tool
# 2.SNR mapping tool
# 3.Signal duration measurements using updated algorithms
# 4.AP and CaT alternan maps
# 5.AP-CaT coupling maps
# 6.Conduction velocity (CV) module
# 7.AP and CaT mapping tool in response to extrasystolic stimulation (S1-S2)
# 8.Region of interest (ROI) analysis

#For inquiries and feedback, please contact us at b26kth@mun.ca. 
#You can find more details about the project in our preprint on bioRxiv: 
 #  https://www.biorxiv.org/content/10.1101/2023.05.01.538926v1.abstract

# Enjoy using Kairosight 3.0 for your research needs!
# Date: 07/14/2023

# Developers : Kazi T. Haq and Anysja Roberts

from analysisfiles import ImagingAnalysis
import cv2
import math
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from numpy import loadtxt, load, save, savetxt
import numpy as np
import os
from PyQt5 import QtCore
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, QObject, QRunnable, Qt, 
                          QThreadPool)
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget
import sys
import time
import traceback
from util.analysis import (act_ind_calc, apd_ind_calc, calc_snr, 
                           calc_tran_activation, cond_vel, diast_ind_calc, 
                           draw_mask, ensemble_xlsx_print, mult_cond_vel, 
                           signal_data_xlsx_print, tau_calc, oap_peak_calc)
from ui.KairoSight_WindowMain_Ret import Ui_MainWindow
from util.preparation import open_stack
from util.processing import filter_drift, filter_spatial_stack, filter_temporal


class JobRunner(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and
    wrap-up.

    :param callback: The function callback to run on this worker thread.
                     Supplied args and kwargs will be passed through to the
                     runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''
    def __init__(self, fn, *args, **kwargs):
        super(JobRunner, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            # Return the result of the processing
            self.signals.result.emit(result)
        finally:
            # Done
            self.signals.finished.emit()

class MainWindow(QWidget, Ui_MainWindow):
    """Customization for Ui_WindowMain"""

    def __init__(self, parent=None, file_purepath=None):
        # Initialization of the superclass
        super(MainWindow, self).__init__(parent)
        '''self.WindowMDI = parent'''
        # Setup the UI
        self.setupUi(self)
        # Save the background color for future reference
        self.bkgd_color = [237/255,237/255,255/255]
        # Setup the image window
        self.mpl_canvas = MplCanvas(self)
        self.mpl_vl_window.addWidget(self.mpl_canvas)
        # Match the matplotlib figure background color to the GUI
        self.mpl_canvas.fig.patch.set_facecolor(self.bkgd_color)
        # Setup the signal windows
        self.mpl_canvas_sig1 = MplCanvas(self)
        self.mpl_sigvl_window.addWidget(self.mpl_canvas_sig1)
        self.mpl_canvas_sig1.fig.patch.set_facecolor(self.bkgd_color)
        self.mpl_canvas_sig2 = MplCanvas(self)
        self.mpl_sigvl_window.addWidget(self.mpl_canvas_sig2)
        self.mpl_canvas_sig2.fig.patch.set_facecolor(self.bkgd_color)
        self.mpl_canvas_sig3 = MplCanvas(self)
        self.mpl_sigvl_window.addWidget(self.mpl_canvas_sig3)
        self.mpl_canvas_sig3.fig.patch.set_facecolor(self.bkgd_color)
        self.mpl_canvas_sig4 = MplCanvas(self)
        self.mpl_sigvl_window.addWidget(self.mpl_canvas_sig4)
        self.mpl_canvas_sig4.fig.patch.set_facecolor(self.bkgd_color)
        # Setup button functionality
        self.sel_dir_button.clicked.connect(self.sel_dir)
        self.load_button.clicked.connect(self.load_data)
        self.refresh_button.clicked.connect(self.refresh_data)
        self.data_prop_button.clicked.connect(self.data_properties)
        self.crop_cb.stateChanged.connect(self.crop_enable)
        self.signal_select_button.clicked.connect(self.signal_select)
        self.reset_signal_button.clicked.connect(self.refresh_signal_selection)
        self.prep_button.clicked.connect(self.run_prep)
        self.analysis_drop.currentIndexChanged.connect(self.analysis_select)
        self.map_pushbutton.clicked.connect(self.map_analysis)
        self.axes_start_time_edit.editingFinished.connect(self.update_win)
        self.axes_end_time_edit.editingFinished.connect(self.update_win)
        self.export_data_button.clicked.connect(self.export_data_numeric)
        self.start_time_edit.editingFinished.connect(self.update_analysis_win)
        self.end_time_edit.editingFinished.connect(self.update_analysis_win)
        self.start_time_edit2.editingFinished.connect(self.update_analysis_win)
        self.end_time_edit2.editingFinished.connect(self.update_analysis_win)
        self.max_val_edit.editingFinished.connect(self.update_analysis_win)
        self.movie_scroll_obj.valueChanged.connect(self.update_axes)
        self.play_movie_button.clicked.connect(self.play_movie)
        self.pause_button.clicked.connect(self.pause_movie)
        self.sig1_x_edit.editingFinished.connect(self.signal_select_edit)
        self.sig1_y_edit.editingFinished.connect(self.signal_select_edit)
        self.sig2_x_edit.editingFinished.connect(self.signal_select_edit)
        self.sig2_y_edit.editingFinished.connect(self.signal_select_edit)
        self.sig3_x_edit.editingFinished.connect(self.signal_select_edit)
        self.sig3_y_edit.editingFinished.connect(self.signal_select_edit)
        self.sig4_x_edit.editingFinished.connect(self.signal_select_edit)
        self.sig4_y_edit.editingFinished.connect(self.signal_select_edit)
        self.export_movie_button.clicked.connect(self.export_movie)
        self.rotate_ccw90_button.clicked.connect(self.rotate_image_ccw90)
        self.rotate_cw90_button.clicked.connect(self.rotate_image_cw90)
        self.crop_xlower_edit.editingFinished.connect(self.crop_update)
        self.crop_xupper_edit.editingFinished.connect(self.crop_update)
        self.crop_ylower_edit.editingFinished.connect(self.crop_update)
        self.crop_yupper_edit.editingFinished.connect(self.crop_update)
        self.colorbar_map_update.clicked.connect(self.map_analysis_cbar)
        self.post_mapping_analysis.clicked.connect(self.post_mapping_data_analysis)
        self.mean_post_mapping_analysis.clicked.connect(self.mean_post_mapping_data_analysis)
        self.mask_draw.clicked.connect(self.draw_mask_area)
        self.mask_use.clicked.connect(self.use_saved_mask_area)
        self.mask_reset.clicked.connect(self.refresh_mask_area)
        self.single_cv.clicked.connect(self.single_vect_cv)
        self.multi_cv.clicked.connect(self.multi_vect_cv)
        self.multi_vector.clicked.connect(self.all_vector)
        # Thread runner
        self.threadpool = QThreadPool()
        self.data = []
        self.data_filt = []
        self.signal_time = []
        self.analysis_bot_lim = False
        self.analysis_top_lim = False
        self.analysis_y_lim = False
        self.sig_disp_bools = [[False, False], [False, False],
                               [False, False], [False, False]]
        self.signal_emit_done = 1
        self.rotate_tracker = 0
        self.preparation_tracker = 0
        self.cnames = ['cornflowerblue', 'gold', 'springgreen', 'lightcoral']
        # Designate that dividing by zero will not generate an error
        np.seterr(divide='ignore', invalid='ignore')

    # Button Functions
    def sel_dir(self):
        # Open dialogue box for selecting the data directory
        self.file_path = QFileDialog.getExistingDirectory(
            self, "Open Directory", os.getcwd(), QFileDialog.ShowDirsOnly)
        # Update list widget with the contents of the selected directory
        self.refresh_data()
        self.mean_post_mapping_analysis.setEnabled(False)
        

    def load_data(self):
        # Grab the selected items name
        self.file_name = self.file_list.currentItem().text()
        # Load the data stack into the UI
        self.video_data_raw = open_stack(
            source=(self.file_path + "/" + self.file_name))
        # Extract the optical data from the stack
        self.data = self.video_data_raw[0]
        n = 5
        self.data = self.data[n:]
        self.data_prop = self.data
        self.im_bkgd = self.data[0]
        # Populate the axes start and end indices
        self.axes_start_ind = 0
        self.axes_end_ind = self.data.shape[0]-1
        # Populate the mask variable
        self.mask = np.ones([self.data.shape[1], self.data.shape[2]],
                            dtype=bool)
        # Reset the signal selection variables
        self.signal_ind = 0
        self.signal_coord = np.zeros((4, 2)).astype(int)
        self.signal_toggle = np.zeros((4, 1))
        self.norm_flag = 0
        # Reset poly select variables
        self.poly_coord = np.zeros((1, 2)).astype(int)
        self.poly_toggle = False
        self.poly_start = False
        # Update the movie window tools with the appropriate values
        self.movie_scroll_obj.setMaximum(self.data.shape[0])
        self.play_bool = 0
        # Reset text edit values
        self.frame_rate_edit.setText('400')
        #self.image_scale_edit.setText('')
        self.start_time_edit.setText('')
        self.end_time_edit.setText('')
        self.max_apd_edit.setText('')
        self.perc_apd_edit_01.setText('')
        self.perc_apd_edit_02.setText('')
        self.axes_end_time_edit.setText('')
        self.axes_start_time_edit.setText('')
        # Update the axes
        self.update_analysis_win()
        # self.update_axes()
        # Enbable Properties Interface
        self.frame_rate_label.setEnabled(True)
        self.frame_rate_edit.setEnabled(True)

        #self.image_scale_edit.setEnabled(True)
        self.data_prop_button.setEnabled(True)
        self.image_type_label.setEnabled(True)
        self.image_type_drop.setEnabled(True)
        self.ec_coupling_label.setEnabled(True)
        self.ec_coupling_cb.setEnabled(True)
        self.rotate_label.setEnabled(True)
        self.rotate_ccw90_button.setEnabled(True)
        self.rotate_cw90_button.setEnabled(True)
        self.crop_cb.setEnabled(True)
        self.crop_cb.setChecked(False)
        self.crop_xlower_edit.setText('0')
        self.crop_xupper_edit.setText(str(self.data.shape[2]-1))
        self.crop_xbound = [0, self.data.shape[2]-1]
        self.crop_ylower_edit.setText('0')
        self.crop_yupper_edit.setText(str(self.data.shape[1]-1))
        self.crop_ybound = [0, self.data.shape[1]-1]
        # Enable signal coordinate tools and clear edit boxes
        self.sig1_x_edit.setEnabled(False)
        self.sig1_x_edit.setText('')
        self.sig2_x_edit.setEnabled(False)
        self.sig2_x_edit.setText('')
        self.sig3_x_edit.setEnabled(False)
        self.sig3_x_edit.setText('')
        self.sig4_x_edit.setEnabled(False)
        self.sig4_x_edit.setText('')
        self.sig1_y_edit.setEnabled(False)
        self.sig1_y_edit.setText('')
        self.sig2_y_edit.setEnabled(False)
        self.sig2_y_edit.setText('')
        self.sig3_y_edit.setEnabled(False)
        self.sig3_y_edit.setText('')
        self.sig4_y_edit.setEnabled(False)
        self.sig4_y_edit.setText('')
        # Disable Preparation Tools
        self.bin_checkbox.setEnabled(False)
        self.bin_drop.setEnabled(False)
        self.filter_checkbox.setEnabled(False)
        self.filter_label_separator.setEnabled(False)
        self.filter_upper_label.setEnabled(False)
        self.filter_upper_edit.setEnabled(False)
        self.drift_checkbox.setEnabled(False)
        self.drift_drop.setEnabled(False)
        self.normalize_checkbox.setEnabled(False)
        self.prep_button.setEnabled(False)
        # Change the button string
        self.data_prop_button.setText('Save Properties')
        # Disable Analysis Tools
        self.analysis_drop.setEnabled(False)
        self.analysis_drop.setCurrentIndex(0)
        self.start_time_label.setEnabled(False)
        self.start_time_edit.setEnabled(False)
        self.end_time_label.setEnabled(False)
        self.end_time_edit.setEnabled(False)
        self.map_pushbutton.setEnabled(False)
        self.max_apd_label.setEnabled(False)
        self.max_apd_edit.setEnabled(False)
        self.max_val_label.setEnabled(False)
        self.max_val_edit.setEnabled(False)
        self.perc_apd_label_01.setEnabled(False)
        self.perc_apd_edit_01.setEnabled(False)
        self.perc_apd_label_02.setEnabled(False)
        self.perc_apd_edit_02.setEnabled(False)
        # Check the check box
        for n in np.arange(1, len(self.signal_toggle)):
            checkboxname = 'ensemble_cb_0{}'.format(n)
            checkbox = getattr(self, checkboxname)
            checkbox.setChecked(False)
            checkbox.setEnabled(False)
        # Disable Movie and Signal Tools
        self.signal_select_button.setEnabled(False)
        self.movie_scroll_obj.setEnabled(False)
        self.play_movie_button.setEnabled(False)
        self.export_movie_button.setEnabled(False)
        # Disable axes controls and export buttons
        self.axes_start_time_label.setEnabled(False)
        self.axes_start_time_edit.setEnabled(False)
        self.axes_end_time_label.setEnabled(False)
        self.axes_end_time_edit.setEnabled(False)
        self.export_data_button.setEnabled(False)

    def refresh_data(self):
        # Grab the applicable file names of the directory and display
        self.data_files = []
        for file in os.listdir(self.file_path):
            if file.endswith(".tif"):
                self.data_files.append(file)
        # If tif files were identified update the list and button availability
        if len(self.data_files) > 0:
            # Clear any potential items from the list widget
            self.file_list.clear()
            # Populate the list widget with the file names
            self.file_list.addItems(self.data_files)
            # Set the current row to the first (i.e., index = 0)
            self.file_list.setCurrentRow(0)
            # Enable the load and refresh buttons
            self.load_button.setEnabled(True)
            self.refresh_button.setEnabled(True)
        else:
            # Clear any potential items from the list widget
            self.file_list.clear()
            # Disable the load button
            self.load_button.setEnabled(False)
            # Create a message box to communicate the absence of data
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No *.tif files in selected directory.")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    def data_properties(self):
        #defining _translate
        _translate = QtCore.QCoreApplication.translate
        if self.data_prop_button.text() == 'Save Properties':
            #set mask draw button to true
            self.mask_draw.setEnabled(True)
            self.mask_reset.setEnabled(True)
            
            # Populate global variables with frame rate and scale values
            self.data_fps = float(self.frame_rate_edit.text())
            #self.data_scale = float(self.image_scale_edit.text())
            # Check data type and flip if necessary
            if self.image_type_drop.currentIndex() == 0:
                if self.ec_coupling_cb.isChecked() == False:
                    #resetting the dropdown and adding the amount of items we need
                    self.analysis_drop.clear()
                    self.analysis_drop.addItems(["", "", "", "", "", "", "", "", ""])
                    # Membrane potential, flip the data
                    self.data_prop = self.data.astype(float)*-1
                
                    #setting the voltage drop down list
                    self.analysis_drop.setItemText(0, 
                                                   _translate("MainWindow", 
                                                              "Activation"))
                    self.analysis_drop.setItemText(1, _translate("MainWindow", 
                                                                 "APD"))
                    self.analysis_drop.setItemText(2, 
                                                   _translate("MainWindow", 
                                                              "APD Ensemble"))
                    self.analysis_drop.setItemText(3, 
                                                   _translate(
                                                       "MainWindow", 
                                                       "Fixed APD Alternan"))
                    self.analysis_drop.setItemText(4, 
                                                   _translate(
                                                       "MainWindow",
                                                       "Dynamic Alternan"))
                    self.analysis_drop.setItemText(5, 
                                                   _translate(
                                                       "MainWindow", 
                                                       "Manual Alternan"))
                    self.analysis_drop.setItemText(6, _translate("MainWindow", 
                                                                 "S1-S2"))
                    self.analysis_drop.setItemText(7, _translate("MainWindow", 
                                                                 "SNR"))
                    self.analysis_drop.setItemText(8, 
                                                   _translate("MainWindow", 
                                                              "Repolarization"))
                    self.max_apd_label.setText(_translate("MainWindow", 
                                                          "Max APD:    "))
                    self.perc_apd_label_01.setText(
                        _translate("MainWindow", "<html><head/><body><p>% \
                                   APD<span style=\"vertical-align:sub;\">1\
                                       </span>:</p></body></html>"))
                    self.perc_apd_label_02.setText(
                        _translate("MainWindow", "<html><head/><body><p>% \
                                   APD<sub>2</sub>:</p></body></html>"))
                
                elif self.ec_coupling_cb.isChecked() == True:
                    #resetting the dropdown and adding the amoutn of items we need
                    self.analysis_drop.clear()
                    self.analysis_drop.addItem("")
                    self.analysis_drop.addItem("")
                    self.analysis_drop.setItemText(0, 
                                                   _translate("MainWindow", 
                                                              "Activation EC"))
                    self.analysis_drop.setItemText(1, 
                                                   _translate("MainWindow", 
                                                              "Repolarization \
                                                                  EC Latency"))
                    # Membrane potential, flip the data
                    self.data_prop = self.data.astype(float)*-1

            elif self.image_type_drop.currentIndex() == 1:
                #resetting the dropdown and adding the amount of items we need
                self.analysis_drop.clear()
                self.analysis_drop.addItems(["", "", "", "", "", "", "", "", ""])
                
                # Calcium transient, don't flip the data
                self.data_prop = self.data.astype(float)
                
                self.analysis_drop.setItemText(0, _translate("MainWindow", 
                                                         "Activation"))
                self.analysis_drop.setItemText(1, _translate("MainWindow", 
                                                         "CaD"))
                self.analysis_drop.setItemText(2, _translate("MainWindow", 
                                                         "CaD Ensemble"))
                self.analysis_drop.setItemText(3, _translate("MainWindow", 
                                                             "Fixed APD Alternan"))
                
                #note all the functionality is disabled (in code below when:
                #self.analysis_drop.currentIndex() == 1 and 
                #self.image_type_drop.currentIndex() == 1), when user selects 
                #this option with calcium images
                
                #changing the 3rd item to grey, so it's apparent it isn't selectable
                self.analysis_drop.setItemData(3, QColor(Qt.gray), Qt.TextColorRole)
                #creating a tooltip the user can hover over, to understand why they can't select this option
                tooltip = "The Fixed Alternan feature is only available for Voltage images."
                self.analysis_drop.setItemData(3, tooltip, Qt.ToolTipRole)
                
                self.analysis_drop.setItemText(4, _translate("MainWindow",
                                                             "Dynamic Alternan"))
                self.analysis_drop.setItemText(5, _translate("MainWindow",
                                                             "Manual Alternan"))
                self.analysis_drop.setItemText(6, _translate("MainWindow",
                                                             "S1-S2"))
                self.analysis_drop.setItemText(7, _translate("MainWindow", 
                                                         "SNR"))
                self.analysis_drop.setItemText(8, _translate("MainWindow", 
                                                         "Repolarization"))
                self.max_apd_label.setText(_translate("MainWindow", 
                                                      "Max CaD:    "))
                self.perc_apd_label_01.setText(
                    _translate("MainWindow", "<html><head/><body><p>% \
                               CaD<span style=\"vertical-align:sub;\">1\
                                   </span>:</p></body></html>"))
                self.perc_apd_label_02.setText(
                    _translate("MainWindow", "<html><head/><body><p>% \
                               CaD<sub>2</sub>:</p></body></html>"))
                
                if self.ec_coupling_cb.isChecked() == True:
                    #resetting the dropdown and adding the amoutn of items we need
                    self.analysis_drop.clear()
                    self.analysis_drop.addItem("")
                    self.analysis_drop.addItem("")
                    self.analysis_drop.setItemText(0, 
                                                   _translate("MainWindow", 
                                                              "Activation EC"))
                    self.analysis_drop.setItemText(1, 
                                                   _translate("MainWindow", 
                                                              "Repolarization\
                                                                  EC Latency"))
                    # Calcium transient, don't flip the data
                    self.data_prop = self.data.astype(float)
                    
            # Create time vector
            self.signal_time = np.arange(self.data.shape[0])*1/self.data_fps
            # Populate the axes start and end edit boxes
            self.axes_start_time_edit.setText(
                str(self.signal_time[self.axes_start_ind]))
            self.axes_end_time_edit.setText(
                str(self.signal_time[self.axes_end_ind-1]))
            # Adjust the x-axis labeling for the signal windows
            for n in np.arange(1, len(self.signal_coord)+1):
                canvasname = 'mpl_canvas_sig{}'.format(n)
                canvas = getattr(self, canvasname)
                canvas.axes.set_xlim(self.signal_time[0], self.signal_time[-1])
                canvas.fig.tight_layout()
                canvas.draw()
                
            # Activate Movie and Signal Tools
            self.signal_select_button.setEnabled(True)
            self.bin_checkbox.setEnabled(True)
            self.bin_drop.setEnabled(True)
            self.filter_checkbox.setEnabled(True)
            self.filter_label_separator.setEnabled(True)
            self.filter_upper_label.setEnabled(True)
            self.filter_upper_edit.setEnabled(True)
            self.drift_checkbox.setEnabled(True)
            self.drift_drop.setEnabled(True)
            self.normalize_checkbox.setEnabled(True)
            self.prep_button.setEnabled(True)
            # Activate Analysis Tools
            self.analysis_drop.setEnabled(True)
            self.start_time_label.setEnabled(True)
            self.start_time_edit.setEnabled(True)
            self.end_time_label.setEnabled(True)
            self.end_time_edit.setEnabled(True)
            self.map_pushbutton.setEnabled(True)
            
            #settings the objects for Activation mapping and SNR mapping
            if (self.analysis_drop.currentIndex() == 0 or 
                self.analysis_drop.currentIndex() == 7):
                # Disable the APD tools
                self.max_apd_label.setEnabled(False)
                self.max_apd_edit.setEnabled(False)
                self.max_apd_edit.setText('')
                self.max_val_label.setEnabled(False)
                self.max_val_edit.setEnabled(False)
                self.max_val_edit.setText('')
                self.perc_apd_label_01.setEnabled(False)
                self.perc_apd_edit_01.setEnabled(False)
                self.perc_apd_label_02.setEnabled(False)
                self.perc_apd_edit_02.setEnabled(False)
                self.perc_apd_edit_01.setText('')
                self.perc_apd_edit_02.setText('')
                self.analysis_y_lim = False
                self.ensemble_cb_01.setEnabled(False)
                self.ensemble_cb_02.setEnabled(False)
                self.ensemble_cb_03.setEnabled(False)
                self.ensemble_cb_04.setEnabled(False)
                self.start_time_label.setText(_translate("MainWindow", 
                                                         "Start Time: "))
                self.start_time_label.setEnabled(True)
                self.start_time_edit.setEnabled(True)
                self.end_time_label.setText(_translate("MainWindow", 
                                                       "End Time: "))
                self.end_time_label.setEnabled(True)
                self.end_time_edit.setEnabled(True)
                self.start_time_label2.hide()
                self.start_time_edit2.hide()
                self.start_time_edit2.setEnabled(False)
                self.end_time_label2.hide()
                self.end_time_edit2.hide()
                self.end_time_edit2.setEnabled(False)
                
                self.image_scale_label.setEnabled(True)
                self.image_scale_edit.setEnabled(True)
                self.image_scale_edit.setText('')
                self.single_cv.setEnabled(True)
                self.multi_cv.setEnabled(True)
                self.s_vector_no.setEnabled(True)
                self.s_vector_no.setText('')
                self.label_7.setEnabled(True)
                self.m_vector_no.setEnabled(True)
                self.label_9.setEnabled(True)
                self.multi_vector.setEnabled(True)

            #setting the objects for APD New mapping
            elif self.analysis_drop.currentIndex() == 1:
                self.max_apd_label.setEnabled(False)
                self.max_apd_edit.setEnabled(False)
                self.max_apd_edit.setText('')
                self.max_val_label.setEnabled(False)
                self.max_val_edit.setEnabled(False)
                self.max_val_edit.setText('')
                self.perc_apd_label_01.setEnabled(False)
                self.perc_apd_edit_01.setEnabled(False)
                self.perc_apd_label_02.setEnabled(False)
                self.perc_apd_edit_02.setEnabled(False)
                self.perc_apd_edit_01.setText('')
                self.perc_apd_edit_02.setText('')
                self.analysis_y_lim = False
                self.ensemble_cb_01.setEnabled(False)
                self.ensemble_cb_02.setEnabled(False)
                self.ensemble_cb_03.setEnabled(False)
                self.ensemble_cb_04.setEnabled(False)
                self.start_time_label.setText(_translate("MainWindow", 
                                                         "Start Time: "))
                self.start_time_label.setEnabled(True)
                self.start_time_edit.setEnabled(True)
                self.end_time_label.setText(_translate("MainWindow", 
                                                       "End Time: "))
                self.end_time_label.setEnabled(True)
                self.end_time_edit.setEnabled(True)
                self.start_time_label2.hide()
                self.start_time_edit2.hide()
                self.start_time_edit2.setEnabled(False)
                self.end_time_label2.hide()
                self.end_time_edit2.hide()
                self.end_time_edit2.setEnabled(False)
                if self.image_type_drop.currentIndex() == 0:
                    self.perc_apd_label_01.setText(
                        _translate("MainWindow", "<html><head/><body><p>% \
                                   APD<span style=\"vertical-align:sub;\">1\
                                       </span>:</p></body></html>"))
                elif self.image_type_drop.currentIndex() == 1:
                    self.perc_apd_label_01.setText(
                        _translate("MainWindow", "<html><head/><body><p>% \
                                   CaD<span style=\"vertical-align:sub;\">\
                                       1</span>:</p></body></html>"))
                self.perc_apd_label_01.setEnabled(True)
                self.perc_apd_edit_01.setEnabled(True)
                
                self.image_scale_label.setEnabled(False)
                self.image_scale_edit.setEnabled(False)
                self.image_scale_edit.setText('')
                self.single_cv.setEnabled(False)
                self.multi_cv.setEnabled(False)
                self.s_vector_no.setEnabled(False)
                self.s_vector_no.setText('')
                self.label_7.setEnabled(False)
                self.m_vector_no.setEnabled(False)
                self.label_9.setEnabled(False)
                self.multi_vector.setEnabled(False)
                
            #setting the objects for APD Ensemble measurements
            elif self.analysis_drop.currentIndex() == 2:
                self.max_apd_label.setEnabled(False)
                self.max_apd_edit.setEnabled(False)
                self.max_apd_edit.setText('')
                self.max_val_label.setEnabled(True)
                self.max_val_edit.setEnabled(True)
                self.max_val_edit.setText(_translate("MainWindow", "0.7"))
                if self.image_type_drop.currentIndex() == 0:
                    self.perc_apd_label_01.setText(
                        _translate("MainWindow", "<html><head/><body><p>%\
                                   APD<span style=\"vertical-align:sub;\">1\
                                       </span>:</p></body></html>"))
                elif self.image_type_drop.currentIndex() == 1:
                    self.perc_apd_label_01.setText(
                        _translate("MainWindow", "<html><head/><body><p>% \
                                   CaD<span style=\"vertical-align:sub;\">1\
                                       </span>:</p></body></html>"))
                self.perc_apd_label_01.setEnabled(True)
                self.perc_apd_edit_01.setEnabled(True)
                if self.image_type_drop.currentIndex() == 0:
                    self.perc_apd_label_02.setText(
                        _translate("MainWindow", "<html><head/><body><p>% \
                                   APD<sub>2</sub>:</p></body></html>"))
                elif self.image_type_drop.currentIndex() == 1:
                    self.perc_apd_label_02.setText(
                        _translate("MainWindow", "<html><head/><body><p>% \
                                   CaD<sub>2</sub>:</p></body></html>"))
                self.perc_apd_label_02.setEnabled(True)
                self.perc_apd_edit_02.setEnabled(True)
                self.perc_apd_edit_01.setText('')
                self.perc_apd_edit_02.setText('')
                
                # Enable the checkboxes next to populated signal axes
                for cnt, n in enumerate(self.signal_toggle):
                    if n == 1:
                        checkboxname = 'ensemble_cb_0{}'.format(cnt+1)
                        checkbox = getattr(self, checkboxname)
                        checkbox.setEnabled(True)
                self.start_time_label.setText(_translate("MainWindow", 
                                                         "Start Time: "))
                self.start_time_label.setEnabled(True)
                self.start_time_edit.setEnabled(True)
                self.end_time_label.setText(_translate("MainWindow", 
                                                       "End Time: "))
                self.end_time_label.setEnabled(True)
                self.end_time_edit.setEnabled(True)
                self.start_time_label2.hide()
                self.start_time_edit2.hide()
                self.start_time_edit2.setEnabled(False)
                self.end_time_label2.hide()
                self.end_time_edit2.hide()
                self.end_time_edit2.setEnabled(False)
                
                self.image_scale_label.setEnabled(False)
                self.image_scale_edit.setEnabled(False)
                self.image_scale_edit.setText('')
                self.single_cv.setEnabled(False)
                self.multi_cv.setEnabled(False)
                self.s_vector_no.setEnabled(False)
                self.s_vector_no.setText('')
                self.label_7.setEnabled(False)
                self.m_vector_no.setEnabled(False)
                self.label_9.setEnabled(False)
                self.multi_vector.setEnabled(False)
            #setting the objects for Alternan mapping
            
            elif self.analysis_drop.currentIndex() == 3:
                if self.image_type_drop.currentIndex() == 0: 
                    #show the label and text box for this function
                    self.start_time_edit2.show()
                    self.start_time_edit2.setEnabled(True)
                    self.start_time_label2.show()
                    self.start_time_label2.setEnabled(True)
                    self.end_time_edit2.show()
                    self.end_time_edit2.setEnabled(True)
                    self.end_time_label2.show()
                    self.end_time_label2.setEnabled(True)
                    self.start_time_label.setText(_translate("MainWindow", 
                                                             "Alt. 1 Start Time: "))
                    self.end_time_label.setText(_translate("MainWindow", 
                                                           "Alt. 1 End Time: "))

                    self.perc_apd_label_01.setText(
                        _translate("MainWindow", "<html><head/><body><p>% APD<span \
                                   style=\"vertical-align:sub;\">1</span>:</p></body>\
                                       </html>"))

                    self.perc_apd_label_01.setEnabled(True)
                    self.perc_apd_edit_01.setEnabled(True)
                    self.perc_apd_edit_01.setText('')
                    
                    self.max_val_label.setEnabled(False)
                    self.max_val_edit.setEnabled(False)
                    self.max_val_edit.setText('')
                    
                    self.max_apd_label.setEnabled(False)
                    self.max_apd_edit.setEnabled(False)
                    
                    self.image_scale_label.setEnabled(False)
                    self.image_scale_edit.setEnabled(False)
                    self.image_scale_edit.setText('')
                    self.single_cv.setEnabled(False)
                    self.multi_cv.setEnabled(False)
                    self.s_vector_no.setEnabled(False)
                    self.s_vector_no.setText('')
                    self.label_7.setEnabled(False)
                    self.m_vector_no.setEnabled(False)
                    self.label_9.setEnabled(False)
                    self.multi_vector.setEnabled(False)
                elif self.image_type_drop.currentIndex() == 1:
                    #make all the funcionality disabled, so the user cannot run 
                    #fixed alternan on calcium images
                    self.start_time_edit2.show()
                    self.start_time_edit2.setEnabled(False)
                    self.start_time_label2.show()
                    self.start_time_label2.setEnabled(False)
                    self.end_time_edit2.show()
                    self.end_time_edit2.setEnabled(False)
                    self.end_time_label2.show()
                    self.end_time_label2.setEnabled(False)
                    self.start_time_label.setText(_translate("MainWindow", 
                                                             "Alt. 1 Start Time: "))
                    self.start_time_label.setEnabled(False)
                    self.start_time_edit.setEnabled(False)
                    self.end_time_label.setText(_translate("MainWindow", 
                                                           "Alt. 1 End Time: "))
                    self.end_time_label.setEnabled(False)
                    self.end_time_edit.setEnabled(False)
                    self.perc_apd_label_01.setText(
                        _translate("MainWindow", "<html><head/><body><p>% APD<span \
                                   style=\"vertical-align:sub;\">1</span>:</p></body>\
                                       </html>"))

                    self.perc_apd_label_01.setEnabled(False)
                    self.perc_apd_edit_01.setEnabled(False)
                    self.perc_apd_edit_01.setText('')
                    self.perc_apd_label_02.setEnabled(False)
                    self.perc_apd_edit_02.setEnabled(False)
                    self.perc_apd_edit_02.setText('')
                    
                    self.max_val_label.setEnabled(False)
                    self.max_val_edit.setEnabled(False)
                    self.max_val_edit.setText('')
                    
                    self.max_apd_label.setEnabled(False)
                    self.max_apd_edit.setEnabled(False)
                    
                    self.image_scale_label.setEnabled(False)
                    self.image_scale_edit.setEnabled(False)
                    self.image_scale_edit.setText('')
                    self.single_cv.setEnabled(False)
                    self.multi_cv.setEnabled(False)
                    self.s_vector_no.setEnabled(False)
                    self.s_vector_no.setText('')
                    self.label_7.setEnabled(False)
                    self.m_vector_no.setEnabled(False)
                    self.label_9.setEnabled(False)
                    self.multi_vector.setEnabled(False)
                
            #setting the objects for dynamic and manual alternan mapping
            elif (self.analysis_drop.currentIndex() == 4 or 
                  self.analysis_drop.currentIndex() == 5):
                self.max_apd_label.setEnabled(False)
                self.max_apd_edit.setEnabled(False)
                self.max_apd_edit.setText('')
                self.max_val_label.setEnabled(False)
                self.max_val_edit.setEnabled(False)
                self.max_val_edit.setText('')
                self.perc_apd_label_01.setEnabled(False)
                self.perc_apd_edit_01.setEnabled(False)
                self.perc_apd_label_02.setEnabled(False)
                self.perc_apd_edit_02.setEnabled(False)
                self.perc_apd_edit_01.setText('')
                self.perc_apd_edit_02.setText('')
                self.analysis_y_lim = False
                self.ensemble_cb_01.setEnabled(False)
                self.ensemble_cb_02.setEnabled(False)
                self.ensemble_cb_03.setEnabled(False)
                self.ensemble_cb_04.setEnabled(False)
                self.start_time_label.setText(_translate("MainWindow", 
                                                         "Start Time: "))
                self.start_time_label.setEnabled(True)
                self.start_time_edit.setEnabled(True)
                self.end_time_label.setText(_translate("MainWindow", 
                                                       "End Time: "))
                self.end_time_label.setEnabled(True)
                self.end_time_edit.setEnabled(True)
                self.start_time_label2.hide()
                self.start_time_edit2.hide()
                self.start_time_edit2.setEnabled(False)
                self.end_time_label2.hide()
                self.end_time_edit2.hide()
                self.end_time_edit2.setEnabled(False)
                self.perc_apd_label_01.setText(_translate("MainWindow", 
                                                          "Peak Coeff."))
                self.perc_apd_label_01.setEnabled(True)
                self.perc_apd_edit_01.setEnabled(True)
                self.perc_apd_edit_01.setText('4')
                
                self.image_scale_label.setEnabled(False)
                self.image_scale_edit.setEnabled(False)
                self.image_scale_edit.setText('')
                self.single_cv.setEnabled(False)
                self.multi_cv.setEnabled(False)
                self.s_vector_no.setEnabled(False)
                self.s_vector_no.setText('')
                self.label_7.setEnabled(False)
                self.m_vector_no.setEnabled(False)
                self.label_9.setEnabled(False)
                self.multi_vector.setEnabled(False)
 
            #S1-S2 Mapping
            elif self.analysis_drop.currentIndex() == 6:
                #show the label and text box for this function
                self.start_time_edit2.show()
                self.start_time_edit2.setEnabled(True)
                self.start_time_label2.show()
                self.start_time_label2.setEnabled(True)
                self.end_time_edit2.show()
                self.end_time_edit2.setEnabled(True)
                self.end_time_label2.show()
                self.end_time_label2.setEnabled(True)
                self.start_time_label.setText(_translate("MainWindow", 
                                                         "Alt. 1 Start Time: "))
                self.start_time_label.setEnabled(True)
                self.start_time_edit.setEnabled(True)
                self.end_time_label.setText(_translate("MainWindow", 
                                                       "Alt. 1 End Time: "))
                self.end_time_label.setEnabled(True)
                self.end_time_edit.setEnabled(True)
                if self.image_type_drop.currentIndex() == 0:
                    self.perc_apd_label_01.setText(
                        _translate("MainWindow", "<html><head/><body><p>% \
                                   APD<span style=\"vertical-align:sub;\">1\
                                       </span>:</p></body></html>"))
                elif self.image_type_drop.currentIndex() == 1:
                    self.perc_apd_label_01.setText(
                        _translate("MainWindow", "<html><head/><body><p>%\
                                   CaD<span style=\"vertical-align:sub;\">1\
                                       </span>:</p></body></html>"))
                self.perc_apd_label_01.setEnabled(True)
                self.perc_apd_edit_01.setEnabled(True)
                self.perc_apd_label_02.setText(_translate("MainWindow", 
                                                          "Peak Coeff."))
                self.perc_apd_label_02.setEnabled(True)
                self.perc_apd_edit_02.setEnabled(True)
                self.perc_apd_edit_01.setText('')
                self.perc_apd_edit_02.setText('4')   
                
                #creating an option for the threshold, but using the max amp text box
                self.max_val_label.setEnabled(True)
                self.max_val_edit.setEnabled(True)
                self.max_val_edit.setText(_translate("MainWindow", "0.7"))
                
                self.max_apd_label.setEnabled(False)
                self.max_apd_edit.setEnabled(False)
                
                self.image_scale_label.setEnabled(False)
                self.image_scale_edit.setEnabled(False)
                self.image_scale_edit.setText('')
                self.single_cv.setEnabled(False)
                self.multi_cv.setEnabled(False)
                self.s_vector_no.setEnabled(False)
                self.s_vector_no.setText('')
                self.label_7.setEnabled(False)
                self.m_vector_no.setEnabled(False)
                self.label_9.setEnabled(False)
                self.multi_vector.setEnabled(False)
                
            #setting the objects for Repolarization mapping 
            elif self.analysis_drop.currentIndex() == 8: 
                # Enable the APD tools 
                self.max_apd_label.setEnabled(True) 
                self.max_apd_edit.setEnabled(True) 
                if self.image_type_drop.currentIndex() == 0:
                    self.perc_apd_label_01.setText(
                        _translate("MainWindow", "<html><head/><body><p>% \
                                   APD<span style=\"vertical-align:sub;\">1\
                                       </span>:</p></body></html>"))
                elif self.image_type_drop.currentIndex() == 1:
                    self.perc_apd_label_01.setText(
                        _translate("MainWindow", "<html><head/><body><p>% \
                                   CaD<span style=\"vertical-align:sub;\">1\
                                       </span>:</p></body></html>"))
                self.perc_apd_label_01.setEnabled(True) 
                self.perc_apd_edit_01.setEnabled(True) 
                self.perc_apd_edit_01.setText('')
                self.perc_apd_label_02.setEnabled(False) 
                self.perc_apd_edit_02.setEnabled(False) 
                # Disable amplitude and checkboxes 
                self.max_val_label.setEnabled(False) 
                self.max_val_edit.setEnabled(False) 
                self.max_val_edit.setText('') 
                self.analysis_y_lim = False 
                self.ensemble_cb_01.setEnabled(False) 
                self.ensemble_cb_02.setEnabled(False) 
                self.ensemble_cb_03.setEnabled(False) 
                self.ensemble_cb_04.setEnabled(False) 
                self.start_time_label.setText(_translate("MainWindow", 
                                                         "Start Time: ")) 
                self.start_time_label.setEnabled(True)
                self.start_time_edit.setEnabled(True)
                self.end_time_label.setText(_translate("MainWindow", 
                                                       "End Time: ")) 
                self.end_time_label.setEnabled(True)
                self.end_time_edit.setEnabled(True)
                self.start_time_label2.hide() 
                self.start_time_edit2.hide() 
                self.start_time_edit2.setEnabled(False) 
                self.end_time_label2.hide() 
                self.end_time_edit2.hide() 
                self.end_time_edit2.setEnabled(False) 

                self.image_scale_label.setEnabled(False) 
                self.image_scale_edit.setEnabled(False) 
                self.image_scale_edit.setText('') 
                self.single_cv.setEnabled(False) 
                self.multi_cv.setEnabled(False) 
                self.s_vector_no.setEnabled(False) 
                self.s_vector_no.setText('') 
                self.label_7.setEnabled(False) 
                self.m_vector_no.setEnabled(False) 
                self.label_9.setEnabled(False) 
                self.multi_vector.setEnabled(False) 
                
            # Activate axes controls
            self.axes_start_time_label.setEnabled(True)
            self.axes_start_time_edit.setEnabled(True)
            self.axes_end_time_label.setEnabled(True)
            self.axes_end_time_edit.setEnabled(True)
            # Activate axes signal selection edit boxes
            axes_on = int(sum(self.signal_toggle)+1)
            for cnt in np.arange(axes_on):
                if cnt == 4:
                    continue
                else:
                    xname = 'sig{}_x_edit'.format(cnt+1)
                    x = getattr(self, xname)
                    x.setEnabled(True)
                    yname = 'sig{}_y_edit'.format(cnt+1)
                    y = getattr(self, yname)
                    y.setEnabled(True)
                    
            # Disable Properties Tools
            self.frame_rate_label.setEnabled(False)
            self.frame_rate_edit.setEnabled(False)

            self.image_type_label.setEnabled(False)
            self.image_type_drop.setEnabled(False)
            self.ec_coupling_label.setEnabled(False)
            self.ec_coupling_cb.setEnabled(False)
            self.rotate_label.setEnabled(False)
            self.rotate_ccw90_button.setEnabled(False)
            self.rotate_cw90_button.setEnabled(False)
            # Check for image crop
            if self.crop_cb.isChecked():
                self.data_prop = self.data_prop[:,
                                                self.crop_ybound[0]:
                                                    self.crop_ybound[1]+1,
                                                self.crop_xbound[0]:
                                                    self.crop_xbound[1]+1]
                save('data_crop.npy', self.data_prop)
                self.im_bkgd = self.im_bkgd[self.crop_ybound[0]:
                                            self.crop_ybound[1]+1,
                                            self.crop_xbound[0]:
                                            self.crop_xbound[1]+1]
            else:
                save('data.npy', self.data_prop)
                
            self.crop_cb.setEnabled(False)
            self.crop_xlower_edit.setEnabled(False)
            self.crop_xupper_edit.setEnabled(False)
            self.crop_ylower_edit.setEnabled(False)
            self.crop_yupper_edit.setEnabled(False)
            # Update preparation tracker
            self.preparation_tracker = 0
            # Change the button string
            self.data_prop_button.setText('Update Properties')
            # Update the axes
            self.update_axes()

        else:
            # Disable Processing Tools
            self.bin_checkbox.setChecked(False)
            self.bin_checkbox.setEnabled(False)
            self.bin_drop.setCurrentIndex(0)
            self.bin_drop.setEnabled(False)
            self.filter_checkbox.setChecked(False)
            self.filter_checkbox.setEnabled(False)
            self.filter_label_separator.setEnabled(False)
            self.filter_upper_label.setEnabled(False)
            self.filter_upper_edit.setEnabled(False)
            self.drift_checkbox.setChecked(False)
            self.drift_checkbox.setEnabled(False)
            self.drift_drop.setCurrentIndex(0)
            self.drift_drop.setEnabled(False)
            self.normalize_checkbox.setChecked(False)
            self.normalize_checkbox.setEnabled(False)
            self.prep_button.setEnabled(False)
            # Disable Analysis Tools
            self.analysis_drop.setEnabled(False)
            self.start_time_label.setEnabled(False)
            self.start_time_edit.setEnabled(False)
            self.start_time_edit.setText('')
            self.start_time_label2.setEnabled(False)
            self.start_time_edit2.setEnabled(False)
            self.start_time_edit.setText('')
            self.end_time_label.setEnabled(False)
            self.end_time_edit.setEnabled(False)
            self.end_time_edit.setText('')
            self.end_time_label2.setEnabled(False)
            self.end_time_edit2.setEnabled(False)
            self.end_time_edit.setText('')
            self.map_pushbutton.setEnabled(False)
            self.max_apd_label.setEnabled(False)
            self.max_apd_edit.setEnabled(False)
            self.max_apd_edit.setText('')
            self.max_val_label.setEnabled(False)
            self.max_val_edit.setEnabled(False)
            self.max_val_edit.setText('')
            self.perc_apd_label_01.setEnabled(False)
            self.perc_apd_edit_01.setEnabled(False)
            self.perc_apd_label_02.setEnabled(False)
            self.perc_apd_edit_02.setEnabled(False)
            self.perc_apd_edit_01.setText('')
            self.perc_apd_edit_02.setText('')
            # Disable Movie and Signal Tools
            self.signal_select_button.setEnabled(False)
            self.movie_scroll_obj.setEnabled(False)
            self.play_movie_button.setEnabled(False)
            self.export_movie_button.setEnabled(False)
            #self.reset_signal_button(False)
            # Disable axes controls
            self.axes_start_time_label.setEnabled(False)
            self.axes_start_time_edit.setEnabled(False)
            self.axes_end_time_label.setEnabled(False)
            self.axes_end_time_edit.setEnabled(False)
            self.sig1_x_edit.setEnabled(False)
            self.sig1_y_edit.setEnabled(False)
            self.sig2_x_edit.setEnabled(False)
            self.sig2_y_edit.setEnabled(False)
            self.sig3_x_edit.setEnabled(False)
            self.sig3_y_edit.setEnabled(False)
            self.sig4_x_edit.setEnabled(False)
            self.sig4_y_edit.setEnabled(False)
            # Reset signal variables
            self.signal_ind = 0
            self.signal_coord = np.zeros((4, 2)).astype(int)
            self.signal_toggle = np.zeros((4, 1))
            self.norm_flag = 0
            self.play_bool = 0
            # Reset analysis tools
            self.start_time_edit.setText('')
            self.end_time_edit.setText('')
            self.max_apd_edit.setText('')
            self.perc_apd_edit_01.setText('')
            self.perc_apd_edit_02.setText('')
            self.axes_end_time_edit.setText('')
            self.axes_start_time_edit.setText('')
            # Activate Properties Tools
            self.frame_rate_label.setEnabled(True)
            self.frame_rate_edit.setEnabled(True)
            self.image_type_label.setEnabled(True)
            self.image_type_drop.setEnabled(True)
            self.ec_coupling_label.setEnabled(True)
            self.ec_coupling_cb.setEnabled(True)
            self.rotate_label.setEnabled(True)
            self.rotate_ccw90_button.setEnabled(True)
            self.rotate_cw90_button.setEnabled(True)
            # Check for image crop
            if self.crop_cb.isChecked():
                self.data = self.video_data_raw[0]
                #removing the first 5 tiffs to get rid of the blip from lights 
                #being turned on
                n = 5
                self.data = self.data[n:]
                self.im_bkgd = self.data[0]
            if self.rotate_tracker != 0:
                self.data = np.rot90(self.data,
                                     k=self.rotate_tracker,
                                     axes=(1, 2))
                #removing the first 5 tiffs to get rid of the blip from lights 
                #being turned on
                n = 5
                self.data = self.data[n:]
                self.im_bkgd = np.rot90(self.im_bkgd,
                                        k=self.rotate_tracker,
                                        axes=(1, 2))
            self.crop_cb.setEnabled(True)
            self.crop_xlower_edit.setEnabled(True)
            self.crop_xupper_edit.setEnabled(True)
            self.crop_ylower_edit.setEnabled(True)
            self.crop_yupper_edit.setEnabled(True)
            # Change the button string
            self.data_prop_button.setText('Save Properties')
            # Update the axes
            self.update_analysis_win()
            self.update_axes()

    def run_prep(self):
        # Pass the function to execute
        runner = JobRunner(self.prep_data)
        # Execute
        self.threadpool.start(runner)
    
    def draw_mask_area(self): 
        gg = draw_mask(self.data_prop[10,:,:]) 
        plt.close('all')
        f = plt.figure()
        sp = f.add_subplot(111)
        tt=gg.astype(int)
        img_mod = cv2.fillPoly(self.data_prop[10,:,:], pts= [tt], color = 
                               (0, 0, 0))
        
        if self.image_type_drop.currentIndex() == 0:
            #AP
            mask_area=np.argwhere(img_mod< 0)
            #setting a variable to know which drop down was selected when the 
            #mask area was saved
            self.drop_down_mask_area = 0
        elif self.image_type_drop.currentIndex() == 1:
            mask_area=np.argwhere(img_mod> 0)
            #setting a variable to know which drop down was selected when the 
            #mask area was saved
            self.drop_down_mask_area = 1
        
        savetxt('img_mod.csv', img_mod, delimiter=',')
        savetxt('gg.csv', gg, delimiter=',')

        self.mask_use.setEnabled(True)
        
        self.data_prop[:,mask_area[:,0],mask_area[:,1]]=0
        self.data_prop[10,:,:]=self.data_prop[11,:,:]
        sp.imshow(self.data_prop[10,:,:])
        plt.show()

        return f, img_mod, mask_area, sp
        
    def use_saved_mask_area(self):
        img_mod = loadtxt('img_mod.csv', delimiter=',')
        f = plt.figure()
        sp = f.add_subplot(111)

        if self.drop_down_mask_area == 0:
            mask_area = np.argwhere(img_mod < 0)
        elif self.drop_down_mask_area == 1:
            mask_area = np.argwhere(img_mod > 0)
        
        self.data_prop[:,mask_area[:,0],mask_area[:,1]]=0
        self.data_prop[10,:,:]=self.data_prop[11,:,:]
        sp.imshow(self.data_prop[10,:,:])
        plt.show()
        
    def refresh_mask_area(self):
        # Grab the selected items name
        self.file_name = self.file_list.currentItem().text()
        # Load the data stack into the UI
        self.video_data_raw = open_stack(
            source=(self.file_path + "/" + self.file_name))
        # Extract the optical data from the stack
        self.data = self.video_data_raw[0]
        #removing the first 5 tiffs to get rid of the blip from lights 
        #being turned on
        n = 5
        self.data = self.data[n:]
        if self.crop_cb.isChecked():
            self.data_prop = load('data_crop.npy')
        else:
            self.data_prop = load('data.npy')
    
    def refresh_signal_selection(self):
        # Reset signal variables
        self.signal_ind = 0
        self.signal_coord = np.zeros((4, 2)).astype(int)
        self.signal_toggle = np.zeros((4, 1))
        self.norm_flag = 0
        self.play_bool = 0
        self.update_axes()
        
    def single_vect_cv (self):
        # Calculate activation
        # Find the time index value to which the start entry is closest
        start_ind = abs(self.signal_time-float(self.start_time_edit.text()))
        start_ind = np.argmin(start_ind)
        # Find the time index value to which the top entry is closest
        end_ind = abs(self.signal_time-float(self.end_time_edit.text()))
        end_ind = np.argmin(end_ind)
        
        act_ind = calc_tran_activation(self.data_filt, start_ind, end_ind)
        act_val = act_ind*(1/self.data_fps)
        scale_factor=float(self.image_scale_edit.text())
        no_v = int(self.s_vector_no.text())
        #for i in range(no_v):
        pts2=cond_vel(act_val,no_v)
        #act=(act_val)
        plt.close('all')
        ff = plt.figure()
        c_fig = ff.add_subplot(111)
        c_fig.imshow(act_val)

        for i in range(no_v):
            plt.arrow(x = pts2[2*i,0], y = pts2[2*i,1],
                      dx=(pts2[2*i+1,0]-pts2[2*i,0]),
                      dy=(pts2[2*i+1,1]-pts2[2*i,1]),
                      width=0.04 ,head_width=3, head_length=3) 
            dxx=(abs((pts2[2*i+1,0] + pts2[2*i,0]))/2).astype(int)-3
            dyy=(abs((pts2[2*i+1,1] + pts2[2*i,1]))/2).astype(int)+3
            vl=np.sqrt(abs((pts2[2*i+1,0]-pts2[2*i,0])**2 + 
                           (pts2[2*i+1,1]-pts2[2*i,1])**2)).astype(int)
            vl=vl/scale_factor
            pt1=act_val[pts2[2*i,1],pts2[2*i,0]]
            pt2=act_val[pts2[2*i+1,1],pts2[2*i+1,0]]
            
            if abs(pt2-pt1) != 0: 
               velocity=(vl/abs(pt2-pt1)).astype(int)  
            else: 
               velocity=0
            plt.annotate(velocity,xy=(dxx,dyy), ha='center', 
                         va='center', size=11, color='red')
        return ff
       
    def multi_vect_cv(self):
        # Calculate activation
        # Find the time index value to which the start entry is closest
        start_ind = abs(self.signal_time-float(self.start_time_edit.text()))
        start_ind = np.argmin(start_ind)
        # Find the time index value to which the top entry is closest
        end_ind = abs(self.signal_time-float(self.end_time_edit.text()))
        end_ind = np.argmin(end_ind)
        
        act_ind = calc_tran_activation(self.data_filt, start_ind, end_ind)
        act_val = act_ind*(1/self.data_fps)
        scale_factor=float(self.image_scale_edit.text())
        pt=mult_cond_vel(act_val)
        plt.close('all')
        no_v = int(self.m_vector_no.text())
        print(pt)
        stepSize = 360/no_v
        b=pt[0,1]-pt[1,1]
        a=pt[0,0]-pt[1,0]
        pos = []
        t = 0 
        print(no_v)
        rr=np.sqrt(a**2+b**2)
        
        while t < 360: 
            theta=((np.arctan(abs(b/a))*180)/np.pi).astype(int)
            gamma=theta-t
            gamma=np.asarray(gamma).astype(int)
            x2=rr*math.cos(math.radians(gamma))
            y2=rr*math.sin(math.radians(gamma))
            pos.append((x2, y2))
            t += stepSize
        
        y_sz=np.shape(act_val)[0]
        x_sz=np.shape(act_val)[1]
    
        pos=np.asarray(pos).astype(float) 

        for i in range(no_v): 
            pos[i,0]=pos[i,0]+pt[0,0] 
            pos[i,1]=pos[i,1]+pt[0,1] 
            if pos[i,0]==0:
                pos[i,0]=1.0    
            if pos [i,1]==0:
                pos[i,1]=1.0
            if pos[i,0]==x_sz:
                pos[i,0]=x_sz-1    
            if pos [i,1]==y_sz:
                pos[i,1]=y_sz-1

        for i in range(no_v):
            xp=pos[i,0]
            yp=pos[i,1]
            if xp > x_sz or yp > y_sz or xp < 0 or yp < 0 : 
                m=(pos[i,1]-pt[0,1])/(pos[i,0]-pt[0,0])
                if (xp>pt[0,0]): 
                    j=x_sz-1
                    for kk in range (x_sz-1): 
                        yy=m*(j-pt[0,0])+pt[0,1]
                        yy=yy.astype(int) 
                        if yy<y_sz and yy >0:
                            ptt=act_val[yy,j] 
                        else:
                            ptt=0
                        j=j-1    
                        if ptt>0: 
                            pos[i,1]=yy
                            pos[i,0]=j 
                            break
                if (pt[0,0]>xp):
                    for j in range (x_sz-1): 
                        yy=m*(j-pt[0,0])+pt[0,1]
                        yy=yy.astype(int)
                        if yy<y_sz and yy >0:
                            ptt=act_val[yy,j] 
                        else:
                            ptt=0   
                        if ptt>0: 
                            pos[i,1]=yy
                            pos[i,0]=j
                            break

        for i in range(no_v):
            xp=pos[i,0]
            yp=pos[i,1]
            if xp < x_sz and xp>0 and yp<y_sz and yp >0: 
                y_pos=pos[i,1].astype(int)
                x_pos=pos[i,0].astype(int)
                ac_v=act_val[y_pos,x_pos]
                if ac_v==0:
                    m=(pos[i,1]-pt[0,1])/(pos[i,0]-pt[0,0])
                    if (xp>pt[0,0]):
                        j=x_sz-1
                        for kk in range (x_sz-1): 
                            yy=m*(j-pt[0,0])+pt[0,1]
                            yy=yy.astype(int) 
                            if yy<y_sz and yy >0:
                                print(j)
                                ptt=act_val[yy,j] 
                            else:
                                ptt=0
                            if ptt>0: 
                                pos[i,1]=yy
                                pos[i,0]=j
                                break
                   
                    if (pt[0,0]>xp):
                        for j in range (x_sz-1): 
                            yy=m*(j-pt[0,0])+pt[0,1]
                            yy=yy.astype(int)
                            if yy<y_sz and yy >0:
                                ptt=act_val[yy,j] 
                            else:
                                ptt=0   
                            if ptt>0: 
                                pos[i,1]=yy
                                pos[i,0]=j
                                break

        aa=np.shape(pos)[0]
        velocity=np.ones((aa,1))
        
        for i in range(no_v):
             vl=np.sqrt(abs((pos[i,0]-pt[0,0])**2 + 
                            (pos[i,1]-pt[0,1])**2)).astype(int)
             vl=vl/scale_factor
             yy=pos[i,1].astype(int)
             xx=pos[i,0].astype(int)
             print(xx)
             print(yy)
             pt2=act_val[yy,xx]
             pt1=act_val[pt[0,1],pt[0,0]] 
             if pt2-pt1 > 0: 
                 velocity[i,0]=(vl/abs(pt2-pt1)).astype(float) 
             else: 
                 velocity[i,0]=0
        #print(velocity)
        
        velocity=velocity/np.max(velocity)
        ff = plt.figure()
        c_fig = ff.add_subplot(111)
        c_fig.imshow(act_val)
        plt.plot(pos[:,0],pos[:,1],'--')
        for i in range(no_v): 
           dx=pos[i,0]-pt[0,0] 
           dy=pos[i,1]-pt[0,1] 
           if velocity[i,0] > 0: 
               plt.arrow(x = pt[0,0], y=pt[0,1],
                         dx = 0.6*dx*velocity[i,0],
                         dy = 0.6*dy*velocity[i,0],
                         width = 0.04, head_width = 2, head_length = 2)
           
    def all_vector(self):
        # Calculate activation
        # Find the time index value to which the start entry is closest
        start_ind = abs(self.signal_time-float(self.start_time_edit.text()))
        start_ind = np.argmin(start_ind)
        # Find the time index value to which the top entry is closest
        end_ind = abs(self.signal_time-float(self.end_time_edit.text()))
        end_ind = np.argmin(end_ind)
        
        act_ind = calc_tran_activation(self.data_filt,start_ind, end_ind)
        act_val = act_ind*(1/self.data_fps)
   
        img=(act_val)
        img2=img.astype(float)
        row=np.shape(img)[0]
        col=np.shape(img)[1]
        
        velocity=np.zeros((row,col))
        ind=np.zeros((row,col))
 
        for i in range(row):
             for  j in range(col):
              if i < row-1 and i > 0:
               if j < col-1 and j > 0:
                 vel=np.zeros((8,1))  
                 if (img2[i+1,j]-img2[i,j]) > 0:
                     vel[0,0]=1/(img2[i+1,j]-img2[i,j])
                 if  (img2[i+1,j+1]-img2[i,j]) >0:   
                     vel[1,0]=1/(img2[i+1,j+1]-img2[i,j])
                 if (img2[i,j+1]-img2[i,j]) >0:
                     vel[2,0]=1/(img2[i,j+1]-img2[i,j])
                 if (img2[i-1,j+1]-img2[i,j]) >0:
                     vel[3,0]=1/(img2[i-1,j+1]-img2[i,j])
                 if (img2[i-1,j]-img2[i,j]) >0:
                     vel[4,0]=1/(img2[i-1,j]-img2[i,j])
                 if (img2[i-1,j-1]-img2[i,j]) >0:
                     vel[5,0]=1/(img2[i-1,j-1]-img2[i,j])
                 if (img2[i,j-1]-img2[i,j]) >0: 
                     vel[6,0]=1/(img2[i,j-1]-img2[i,j])
                 if (img2[i+1,j-1]-img2[i,j]) >0:
                   vel[7,0]=1/(img2[i+1,j-1]-img2[i,j])
                 velocity[i,j]=np.max(vel) 
                 indx=np.where(vel==np.max(vel))
                 idx=np.asarray(indx).astype(int)
                 ind[i,j]=idx[0,0]
        
        velocity=velocity/np.max(velocity)
        plt.imshow(img)  
        for i in range(row-1):
             for  j in range(col-1):
                 if i < row-1 and i > 0:
                     if j < col-1 and j > 0:
                         if ind[i,j]==0:
                             plt.arrow(x=j,y=i,dx=0,dy=velocity[i,j],
                                       width=0.0005,head_width=0.1,
                                       head_length=0.1)
                         if ind[i,j]==1:
                             plt.arrow(x=j,y=i,
                                       dx=velocity[i,j], dy=velocity[i,j],
                                       width=0.0005,head_width=0.1,
                                       head_length=0.1)
                         if ind[i,j]==2:
                             plt.arrow(x=j,y=i,dx=velocity[i,j],dy=0,
                                       width=0.0005, head_width=0.1,
                                       head_length=0.1)  
                         if ind[i,j]==3:
                            plt.arrow(x=j,y=i,dx=velocity[i,j],
                                      dy=-velocity[i,j],width=0.0005,
                                      head_width=0.1,head_length=0.1)
                         if ind[i,j]==4:
                            plt.arrow(x=j,y=i,dx=0,dy=-velocity[i,j],
                                      width=0.0005,head_width=0.1,
                                      head_length=0.1)
                         if ind[i,j]==5:
                            plt.arrow(x=j,y=i,dx=-velocity[i,j],
                                      dy=-velocity[i,j], width=0.0005,
                                      head_width=0.1,head_length=0.1)
                         if ind[i,j]==6:
                            plt.arrow(x=j,y=i,dx=-velocity[i,j],
                                      dy=0,width=0.0005, head_width=0.1,
                                      head_length=0.1)
                         if ind[i,j]==7:
                            plt.arrow(x=j,y=i,dx=-velocity[i,j],
                                      dy=velocity[i,j],width=0.0005,
                                      head_width=0.1,head_length=0.1)
    
    def prep_data(self, progress_callback):
        # Designate that dividing by zero will not generate an error
        np.seterr(divide='ignore', invalid='ignore')
        # Grab unprepped data and check data type to flip if necessary
        self.data_filt = self.data_prop
        imaging_analysis = ImagingAnalysis.ImagingAnalysis()

        # Spatial filter
        if self.bin_checkbox.isChecked():
            bin_timestart = time.process_time()
            # Grab the kernel size
            bin_kernel = self.bin_drop.currentText()
            if bin_kernel == '3x3':
                bin_kernel = 3
            elif bin_kernel == '5x5':
                bin_kernel = 5
            elif bin_kernel == '7x7':
                bin_kernel = 7
            elif bin_kernel == '9x9':
                bin_kernel = 9
            elif bin_kernel == '15x15':
                bin_kernel = 15
            elif bin_kernel == '21x21':
                bin_kernel = 21
            else:
                bin_kernel = 31
            # Execute spatial filter with selected kernel size
            self.data_filt = filter_spatial_stack(self.data_filt, bin_kernel)
            bin_timeend = time.process_time()
            print(
                f'Binning Time: {bin_timeend-bin_timestart}')

        # Temporal filter
        if self.filter_checkbox.isChecked():
            filter_timestart = time.process_time()
            # Apply the low pass filter
            self.data_filt = filter_temporal(
                self.data_filt, self.data_fps, filter_order=100,
                freq_cutoff=float(self.filter_upper_edit.text()))
            filter_timeend = time.process_time()
            print(
                f'Filter Time: {filter_timeend-filter_timestart}')

        # Drift Removal
        if self.drift_checkbox.isChecked():
            drift_timestart = time.process_time()
            # Grab drift order from dropdown
            drift_order = self.drift_drop.currentIndex()+1
            # Apply drift removal
            self.data_filt = filter_drift(
                self.data_filt, self.mask, drift_order)
            drift_timeend = time.process_time()
            print(f'Drift Time: {drift_timeend-drift_timestart}')

        # Normalization
        if self.normalize_checkbox.isChecked():
            normalize_timestart = time.process_time()
            aa=np.shape(self.data_filt)
            aa=np.array(aa)
            for i in range(aa[1]):
                for j in range(aa[2]): 
                    self.data_filt[:,i,j]=(self.data_filt[:,i,j]-
                                           np.amin(self.data_filt[:,i,j]))\
                    /(np.amax(self.data_filt[:,i,j])-
                      np.amin(self.data_filt[:,i,j]))
            
            self.norm_flag = 1
            normalize_timeend = time.process_time()
            print(
                f'Normalize Time: {normalize_timeend-normalize_timestart}')
        else:
            # Reset normalization flag
            self.norm_flag = 0
        # Update preparation tracker
        self.preparation_tracker = 1
        # Update axes
        self.update_axes()
        # Make the movie screen controls available if normalization occurred
        if self.normalize_checkbox.isChecked():
            self.movie_scroll_obj.setEnabled(True)
            self.play_movie_button.setEnabled(True)
            self.export_movie_button.setEnabled(True)
            self.interp_label.setEnabled(True)
            self.interp_drop.setEnabled(True)
        
        if self.ec_coupling_cb.isChecked():
            imaging_analysis.ec_coupling_save(self.data_filt, 
                                         self.image_type_drop.currentIndex())
        
    def analysis_select(self):
        #defining _translate
        _translate = QtCore.QCoreApplication.translate
        #settings the objects for Activation mapping and SNR mapping
        if (self.analysis_drop.currentIndex() == 0 or 
            self.analysis_drop.currentIndex() == 7):
            # Disable the APD tools
            self.max_apd_label.setEnabled(False)
            self.max_apd_edit.setEnabled(False)
            self.max_apd_edit.setText('')
            self.max_val_label.setEnabled(False)
            self.max_val_edit.setEnabled(False)
            self.max_val_edit.setText('')
            self.perc_apd_label_01.setEnabled(False)
            self.perc_apd_edit_01.setEnabled(False)
            self.perc_apd_label_02.setEnabled(False)
            self.perc_apd_edit_02.setEnabled(False)
            self.perc_apd_edit_01.setText('')
            self.perc_apd_edit_02.setText('')
            self.analysis_y_lim = False
            self.ensemble_cb_01.setEnabled(False)
            self.ensemble_cb_02.setEnabled(False)
            self.ensemble_cb_03.setEnabled(False)
            self.ensemble_cb_04.setEnabled(False)
            self.start_time_label.setText(_translate("MainWindow", 
                                                     "Start Time: "))
            self.start_time_label.setEnabled(True)
            self.start_time_edit.setEnabled(True)
            self.end_time_label.setText(_translate("MainWindow", 
                                                   "End Time: "))
            self.end_time_label.setEnabled(True)
            self.end_time_edit.setEnabled(True)
            self.start_time_label2.hide()
            self.start_time_edit2.hide()
            self.start_time_edit2.setEnabled(False)
            self.end_time_label2.hide()
            self.end_time_edit2.hide()
            self.end_time_edit2.setEnabled(False)
            
            self.image_scale_label.setEnabled(True)
            self.image_scale_edit.setEnabled(True)
            self.image_scale_edit.setText('')
            self.single_cv.setEnabled(True)
            self.multi_cv.setEnabled(True)
            self.s_vector_no.setEnabled(True)
            self.s_vector_no.setText('')
            self.label_7.setEnabled(True)
            self.m_vector_no.setEnabled(True)
            self.label_9.setEnabled(True)
            self.multi_vector.setEnabled(True)
            
        #setting the objects for APD New mapping
        elif self.analysis_drop.currentIndex() == 1:
            self.max_apd_label.setEnabled(False)
            self.max_apd_edit.setEnabled(False)
            self.max_apd_edit.setText('')
            self.max_val_label.setEnabled(False)
            self.max_val_edit.setEnabled(False)
            self.max_val_edit.setText('')
            self.perc_apd_label_01.setEnabled(False)
            self.perc_apd_edit_01.setEnabled(False)
            self.perc_apd_label_02.setEnabled(False)
            self.perc_apd_edit_02.setEnabled(False)
            self.perc_apd_edit_01.setText('')
            self.perc_apd_edit_02.setText('')
            self.analysis_y_lim = False
            self.ensemble_cb_01.setEnabled(False)
            self.ensemble_cb_02.setEnabled(False)
            self.ensemble_cb_03.setEnabled(False)
            self.ensemble_cb_04.setEnabled(False)
            self.start_time_label.setText(_translate("MainWindow", 
                                                     "Start Time: "))
            self.start_time_label.setEnabled(True)
            self.start_time_edit.setEnabled(True)
            self.end_time_label.setText(_translate("MainWindow", 
                                                   "End Time: "))
            self.end_time_label.setEnabled(True)
            self.end_time_edit.setEnabled(True)
            self.start_time_label2.hide()
            self.start_time_edit2.hide()
            self.start_time_edit2.setEnabled(False)
            self.end_time_label2.hide()
            self.end_time_edit2.hide()
            self.end_time_edit2.setEnabled(False)
            if self.image_type_drop.currentIndex() == 0:
                self.perc_apd_label_01.setText(
                    _translate("MainWindow", "<html><head/><body><p>% \
                               APD<span style=\"vertical-align:sub;\">1\
                                   </span>:</p></body></html>"))
            elif self.image_type_drop.currentIndex() == 1:
                self.perc_apd_label_01.setText(
                    _translate("MainWindow", "<html><head/><body><p>% \
                               CaD<span style=\"vertical-align:sub;\">1\
                                   </span>:</p></body></html>"))
            self.perc_apd_label_01.setEnabled(True)
            self.perc_apd_edit_01.setEnabled(True)
            self.perc_apd_edit_01.setText('')
            
            self.image_scale_label.setEnabled(False)
            self.image_scale_edit.setEnabled(False)
            self.image_scale_edit.setText('')
            self.single_cv.setEnabled(False)
            self.multi_cv.setEnabled(False)
            self.s_vector_no.setEnabled(False)
            self.s_vector_no.setText('')
            self.label_7.setEnabled(False)
            self.m_vector_no.setEnabled(False)
            self.label_9.setEnabled(False)
            self.multi_vector.setEnabled(False)
        #setting the objects for APD Ensemble measurements
        elif self.analysis_drop.currentIndex() == 2:
            self.max_apd_label.setEnabled(False)
            self.max_apd_edit.setEnabled(False)
            self.max_apd_edit.setText('')
            self.max_val_label.setEnabled(True)
            self.max_val_edit.setEnabled(True)
            self.max_val_edit.setText(_translate("MainWindow", "0.7"))
            if self.image_type_drop.currentIndex() == 0:
                self.perc_apd_label_01.setText(
                    _translate("MainWindow", "<html><head/><body><p>% \
                               APD<span style=\"vertical-align:sub;\">\
                                   1</span>:</p></body></html>"))
            elif self.image_type_drop.currentIndex() == 1:
                self.perc_apd_label_01.setText(
                    _translate("MainWindow", "<html><head/><body><p>%\
                               CaD<span style=\"vertical-align:sub;\">1\
                                   </span>:</p></body></html>"))
            self.perc_apd_label_01.setEnabled(True)
            self.perc_apd_edit_01.setEnabled(True)
            if self.image_type_drop.currentIndex() == 0:
                self.perc_apd_label_02.setText(
                    _translate("MainWindow", "<html><head/><body><p>% \
                               APD<sub>2</sub>:</p></body></html>"))
            elif self.image_type_drop.currentIndex() == 1:
                self.perc_apd_label_02.setText(
                    _translate("MainWindow", "<html><head/><body><p>% \
                               CaD<sub>2</sub>:</p></body></html>"))
            self.perc_apd_label_02.setEnabled(True)
            self.perc_apd_edit_02.setEnabled(True)
            self.perc_apd_edit_01.setText('')
            self.perc_apd_edit_02.setText('')
            
            # Enable the checkboxes next to populated signal axes
            for cnt, n in enumerate(self.signal_toggle):
                if n == 1:
                    checkboxname = 'ensemble_cb_0{}'.format(cnt+1)
                    checkbox = getattr(self, checkboxname)
                    checkbox.setEnabled(True)
            self.start_time_label.setText(_translate("MainWindow", 
                                                     "Start Time: "))
            self.start_time_label.setEnabled(True)
            self.start_time_edit.setEnabled(True)
            self.end_time_label.setText(_translate("MainWindow", 
                                                   "End Time: "))
            self.end_time_label.setEnabled(True)
            self.end_time_edit.setEnabled(True)
            self.start_time_label2.hide()
            self.start_time_edit2.hide()
            self.start_time_edit2.setEnabled(False)
            self.end_time_label2.hide()
            self.end_time_edit2.hide()
            self.end_time_edit2.setEnabled(False)
            
            self.image_scale_label.setEnabled(False)
            self.image_scale_edit.setEnabled(False)
            self.image_scale_edit.setText('')
            self.single_cv.setEnabled(False)
            self.multi_cv.setEnabled(False)
            self.s_vector_no.setEnabled(False)
            self.s_vector_no.setText('')
            self.label_7.setEnabled(False)
            self.m_vector_no.setEnabled(False)
            self.label_9.setEnabled(False)
            self.multi_vector.setEnabled(False)
            
        #setting the objects for Alternan mapping
        elif self.analysis_drop.currentIndex() == 3:
            if self.image_type_drop.currentIndex() == 0: 
                #show the label and text box for this function
                self.start_time_edit2.show()
                self.start_time_edit2.setEnabled(True)
                self.start_time_label2.show()
                self.start_time_label2.setEnabled(True)
                self.end_time_edit2.show()
                self.end_time_edit2.setEnabled(True)
                self.end_time_label2.show()
                self.end_time_label2.setEnabled(True)
                self.start_time_label.setText(_translate("MainWindow", 
                                                         "Alt. 1 Start Time: "))
                self.end_time_label.setText(_translate("MainWindow", 
                                                       "Alt. 1 End Time: "))
                self.perc_apd_label_01.setText(
                    _translate("MainWindow", "<html><head/><body><p>% APD<span \
                               style=\"vertical-align:sub;\">1</span>:</p></body>\
                                   </html>"))

                self.perc_apd_label_01.setEnabled(True)
                self.perc_apd_edit_01.setEnabled(True)
                self.perc_apd_edit_01.setText('')
                self.perc_apd_label_02.setEnabled(False)
                self.perc_apd_edit_02.setEnabled(False)
                self.perc_apd_edit_02.setText('')
                
                self.max_val_label.setEnabled(False)
                self.max_val_edit.setEnabled(False)
                self.max_val_edit.setText('')
                
                self.max_apd_label.setEnabled(False)
                self.max_apd_edit.setEnabled(False)
                
                self.image_scale_label.setEnabled(False)
                self.image_scale_edit.setEnabled(False)
                self.image_scale_edit.setText('')
                self.single_cv.setEnabled(False)
                self.multi_cv.setEnabled(False)
                self.s_vector_no.setEnabled(False)
                self.s_vector_no.setText('')
                self.label_7.setEnabled(False)
                self.m_vector_no.setEnabled(False)
                self.label_9.setEnabled(False)
                self.multi_vector.setEnabled(False)
            elif self.image_type_drop.currentIndex() == 1:
                #make all the funcionality disabled, so the user cannot run 
                #fixed alternan on calcium images
                self.start_time_edit2.show()
                self.start_time_edit2.setEnabled(False)
                self.start_time_label2.show()
                self.start_time_label2.setEnabled(False)
                self.end_time_edit2.show()
                self.end_time_edit2.setEnabled(False)
                self.end_time_label2.show()
                self.end_time_label2.setEnabled(False)
                self.start_time_label.setText(_translate("MainWindow", 
                                                         "Alt. 1 Start Time: "))
                self.start_time_label.setEnabled(False)
                self.start_time_edit.setEnabled(False)
                self.end_time_label.setText(_translate("MainWindow", 
                                                       "Alt. 1 End Time: "))
                self.end_time_label.setEnabled(False)
                self.end_time_edit.setEnabled(False)
                self.perc_apd_label_01.setText(
                    _translate("MainWindow", "<html><head/><body><p>% APD<span \
                               style=\"vertical-align:sub;\">1</span>:</p></body>\
                                   </html>"))

                self.perc_apd_label_01.setEnabled(False)
                self.perc_apd_edit_01.setEnabled(False)
                self.perc_apd_edit_01.setText('')
                self.perc_apd_label_02.setEnabled(False)
                self.perc_apd_edit_02.setEnabled(False)
                self.perc_apd_edit_02.setText('')
                
                self.max_val_label.setEnabled(False)
                self.max_val_edit.setEnabled(False)
                self.max_val_edit.setText('')
                
                self.max_apd_label.setEnabled(False)
                self.max_apd_edit.setEnabled(False)
                
                self.image_scale_label.setEnabled(False)
                self.image_scale_edit.setEnabled(False)
                self.image_scale_edit.setText('')
                self.single_cv.setEnabled(False)
                self.multi_cv.setEnabled(False)
                self.s_vector_no.setEnabled(False)
                self.s_vector_no.setText('')
                self.label_7.setEnabled(False)
                self.m_vector_no.setEnabled(False)
                self.label_9.setEnabled(False)
                self.multi_vector.setEnabled(False)
                
            
        #setting the objects for dynamic and manual alternan mapping
        elif (self.analysis_drop.currentIndex() == 4 or 
              self.analysis_drop.currentIndex() == 5):
            self.max_apd_label.setEnabled(False)
            self.max_apd_edit.setEnabled(False)
            self.max_apd_edit.setText('')
            self.max_val_label.setEnabled(False)
            self.max_val_edit.setEnabled(False)
            self.max_val_edit.setText('')
            self.perc_apd_label_01.setEnabled(False)
            self.perc_apd_edit_01.setEnabled(False)
            self.perc_apd_label_02.setEnabled(False)
            self.perc_apd_edit_02.setEnabled(False)
            self.perc_apd_edit_01.setText('')
            self.perc_apd_edit_02.setText('')
            self.analysis_y_lim = False
            self.ensemble_cb_01.setEnabled(False)
            self.ensemble_cb_02.setEnabled(False)
            self.ensemble_cb_03.setEnabled(False)
            self.ensemble_cb_04.setEnabled(False)
            self.start_time_label.setText(_translate("MainWindow", 
                                                     "Start Time: "))
            self.start_time_label.setEnabled(True)
            self.start_time_edit.setEnabled(True)
            self.end_time_label.setText(_translate("MainWindow", 
                                                   "End Time: "))
            self.end_time_label.setEnabled(True)
            self.end_time_edit.setEnabled(True)
            self.start_time_label2.hide()
            self.start_time_edit2.hide()
            self.start_time_edit2.setEnabled(False)
            self.end_time_label2.hide()
            self.end_time_edit2.hide()
            self.end_time_edit2.setEnabled(False)
            self.perc_apd_label_01.setText(_translate("MainWindow", 
                                                      "Peak Coeff."))
            self.perc_apd_label_01.setEnabled(True)
            self.perc_apd_edit_01.setEnabled(True)
            self.perc_apd_edit_01.setText('4')
            
            self.image_scale_label.setEnabled(False)
            self.image_scale_edit.setEnabled(False)
            self.image_scale_edit.setText('')
            self.single_cv.setEnabled(False)
            self.multi_cv.setEnabled(False)
            self.s_vector_no.setEnabled(False)
            self.s_vector_no.setText('')
            self.label_7.setEnabled(False)
            self.m_vector_no.setEnabled(False)
            self.label_9.setEnabled(False)
            self.multi_vector.setEnabled(False)
            
        #setting the options for S1-S2 mapping
        elif self.analysis_drop.currentIndex() == 6:
            #show the label and text box for this function
            self.start_time_edit2.show()
            self.start_time_edit2.setEnabled(True)
            self.start_time_label2.show()
            self.start_time_label2.setEnabled(True)
            self.end_time_edit2.show()
            self.end_time_edit2.setEnabled(True)
            self.end_time_label2.show()
            self.end_time_label2.setEnabled(True)
            self.start_time_label.setText(_translate("MainWindow", 
                                                     "Alt. 1 Start Time: "))
            self.start_time_label.setEnabled(True)
            self.start_time_edit.setEnabled(True)
            self.end_time_label.setText(_translate("MainWindow", 
                                                   "Alt. 1 End Time: "))
            self.end_time_label.setEnabled(True)
            self.end_time_edit.setEnabled(True)
            if self.image_type_drop.currentIndex() == 0:
                self.perc_apd_label_01.setText(
                    _translate("MainWindow", "<html><head/><body><p>% \
                               APD<span style=\"vertical-align:sub;\">1\
                                   </span>:</p></body></html>"))
            elif self.image_type_drop.currentIndex() == 1:
                self.perc_apd_label_01.setText(
                    _translate("MainWindow", "<html><head/><body><p>% \
                               CaD<span style=\"vertical-align:sub;\">1\
                                   </span>:</p></body></html>"))
            self.perc_apd_label_01.setEnabled(True)
            self.perc_apd_edit_01.setEnabled(True)
            self.perc_apd_label_02.setText(_translate("MainWindow", 
                                                      "Peak Coeff."))
            self.perc_apd_label_02.setEnabled(True)
            self.perc_apd_edit_02.setEnabled(True)
            self.perc_apd_edit_01.setText('')
            self.perc_apd_edit_02.setText('4')
            
            #creating an option for the threshold, but using the max amp text box
            self.max_val_label.setEnabled(True)
            self.max_val_edit.setEnabled(True)
            self.max_val_edit.setText(_translate("MainWindow", "0.7"))
            
            self.max_apd_label.setEnabled(False)
            self.max_apd_edit.setEnabled(False)
            self.image_scale_label.setEnabled(False)
            self.image_scale_edit.setEnabled(False)
            self.image_scale_edit.setText('')
            self.single_cv.setEnabled(False)
            self.multi_cv.setEnabled(False)
            self.s_vector_no.setEnabled(False)
            self.s_vector_no.setText('')
            self.label_7.setEnabled(False)
            self.m_vector_no.setEnabled(False)
            self.label_9.setEnabled(False)
            self.multi_vector.setEnabled(False)
            
        #setting the objects for Repolarization mapping
        elif self.analysis_drop.currentIndex() == 8: 
            # Enable the APD tools 
            self.max_apd_label.setEnabled(True) 
            self.max_apd_edit.setEnabled(True) 
            if self.image_type_drop.currentIndex() == 0:
                self.perc_apd_label_01.setText(
                    _translate("MainWindow", "<html><head/><body><p>% \
                               APD<span style=\"vertical-align:sub;\">1\
                                   </span>:</p></body></html>"))
            elif self.image_type_drop.currentIndex() == 1:
                self.perc_apd_label_01.setText(
                    _translate("MainWindow", "<html><head/><body><p>% \
                               CaD<span style=\"vertical-align:sub;\">1\
                                   </span>:</p></body></html>"))
            self.perc_apd_label_01.setEnabled(True) 
            self.perc_apd_edit_01.setEnabled(True) 
            self.perc_apd_label_02.setEnabled(False) 
            self.perc_apd_edit_02.setEnabled(False) 
            # Disable amplitude and checkboxes 
            self.max_val_label.setEnabled(False) 
            self.max_val_edit.setEnabled(False) 
            self.max_val_edit.setText('') 
            self.analysis_y_lim = False 
            self.ensemble_cb_01.setEnabled(False) 
            self.ensemble_cb_02.setEnabled(False) 
            self.ensemble_cb_03.setEnabled(False) 
            self.ensemble_cb_04.setEnabled(False) 
            self.start_time_label.setText(_translate("MainWindow", 
                                                     "Start Time: ")) 
            self.start_time_label.setEnabled(True)
            self.start_time_edit.setEnabled(True)
            self.end_time_label.setText(_translate("MainWindow", 
                                                   "End Time: ")) 
            self.end_time_label.setEnabled(True)
            self.end_time_edit.setEnabled(True)
            self.start_time_label2.hide() 
            self.start_time_edit2.hide() 
            self.start_time_edit2.setEnabled(False) 
            self.end_time_label2.hide() 
            self.end_time_edit2.hide() 
            self.end_time_edit2.setEnabled(False) 
             
            self.image_scale_label.setEnabled(False) 
            self.image_scale_edit.setEnabled(False) 
            self.image_scale_edit.setText('') 
            self.single_cv.setEnabled(False) 
            self.multi_cv.setEnabled(False) 
            self.s_vector_no.setEnabled(False) 
            self.s_vector_no.setText('') 
            self.label_7.setEnabled(False) 
            self.m_vector_no.setEnabled(False) 
            self.label_9.setEnabled(False) 
            self.multi_vector.setEnabled(False) 
        # Update the axes accordingly
        self.update_axes()

    def run_map(self):
        # Pass the function to execute
        runner = JobRunner(self.map_analysis)
        # Execute
        self.threadpool.start(runner)

    def map_analysis(self, progress_callback):
        #Activate Post Mappng Analysis button
        self.post_mapping_analysis.setEnabled(True)
        
        #defining _translate
        #_translate = QtCore.QCoreApplication.translate
        # Grab analysis type
        analysis_type = self.analysis_drop.currentIndex()
        # Grab the start and end times
        start_time = float(self.start_time_edit.text())
        end_time = float(self.end_time_edit.text())
        
        if analysis_type == 3 or analysis_type == 6:
            start_time2 = float(self.start_time_edit2.text())
            # find the time index value to which the second top entry is closest
            start_ind2 = abs(self.signal_time-start_time2)
            start_ind2 = np.argmin(start_ind2)
            
            end_time2 = float(self.end_time_edit2.text())
            # find the time index value to which the second top entry is closest
            end_ind2 = abs(self.signal_time-end_time2)
            end_ind2 = np.argmin(end_ind2)
            
        # Find the time index value to which the start entry is closest
        start_ind = abs(self.signal_time-start_time)
        start_ind = np.argmin(start_ind)
        # Find the time index value to which the top entry is closest
        end_ind = abs(self.signal_time-end_time)
        end_ind = np.argmin(end_ind)
        # Grab masking information
        transp = self.mask
        '''# Calculate activation
        self.act_ind = calc_tran_activation(
            self.data_filt, start_ind, end_ind)
        self.act_val = self.act_ind*(1/self.data_fps)
        max_val = (end_ind-start_ind)*(1/self.data_fps)'''
        
        #define imaging_analysis by calling the ImagingAnalysis class
        imaging_analysis = ImagingAnalysis.ImagingAnalysis()
        
        self.label_2.setEnabled(True)
        self.Cbar_min.setEnabled(True)
        self.label_3.setEnabled(True)
        self.Cbar_max.setEnabled(True)
        self.colorbar_map_update.setEnabled(True)
        
        # Generate activation map
        # Calculate the APD 80
        
        if analysis_type == 1:
            if self.ec_coupling_cb.isChecked() == False:
                print('bokchod')
                imaging_analysis = ImagingAnalysis.ImagingAnalysis()
                li1 = "no preset min"
                li2 = "no preset max"
                interp_selection = self.interp_drop.currentIndex()        
                start_time = float(self.start_time_edit.text())
                end_time = float(self.end_time_edit.text())
                apd_input = float(self.perc_apd_edit_01.text())/100
            
            #getting the file path the user selected
                file_path = str(self.file_path)
            #splitting the file path into an array by the /
                file_path_obj = file_path.split('/')
            #getting the length of this object
                length = len(file_path_obj)
            #grabbing the last object of the array, should be the file name the 
            #user selected
                file_id = file_path_obj[length-1]

                mapp2 = imaging_analysis.apd_analysis(self.data_fps, 
                                                      self.data_filt, start_ind, 
                                                      end_ind, interp_selection,
                                                      apd_input, file_id)[0]
                imaging_analysis.imaging_mapping(mapp2, li1, li2, transp)        
        
        
        
        if analysis_type == 0:
            if self.ec_coupling_cb.isChecked() == False:
                #np.savetxt('act.txt',self.data_filt)
                # Calculate activation
                self.act_ind = calc_tran_activation(
                    self.data_filt, start_ind, end_ind)
                self.act_val = self.act_ind*(1/self.data_fps)
                max_val = (end_ind-start_ind)*(1/self.data_fps)
                # Generate a map of the activation times
                self.act_map = plt.figure()
                axes_act_map = self.act_map.add_axes([0.05, 0.1, 0.8, 0.8])
                transp = transp.astype(float)
           
                axes_act_map.imshow(self.act_val, alpha=transp, vmin=0,
                                    vmax=max_val, cmap='jet')
                
               
                cax = plt.axes([0.87, 0.12, 0.05, 0.76])
                self.act_map.colorbar(
                    cm.ScalarMappable(
                        colors.Normalize(0, max_val),
                        cmap='jet'),
                    cax=cax, format='%.3f')
                
                #getting the file path the user selected
                file_path = str(self.file_path)
                #splitting the file path into an array by the /
                file_path_obj = file_path.split('/')
                #getting the length of this object
                length = len(file_path_obj)
                #grabbing the last object of the array, should be the file name the 
                #user selected
                file_id = file_path_obj[length-1]
                
                #making a folder if there isn't a "Saved Data Maps" folder
                if not os.path.exists("Saved Data Maps\\" + file_id):
                   os.makedirs("Saved Data Maps\\" + file_id)
                   
                #saving the data file
                savetxt('Saved Data Maps\\' + file_id + '\\activation.csv', 
                        self.act_val, delimiter=',')
                
        if self.ec_coupling_cb.isChecked() == True: 
                #getting the file path the user selected
                file_path = str(self.file_path)
                #splitting the file path into an array by the /
                file_path_obj = file_path.split('/')
                #getting the length of this object
                length = len(file_path_obj)
                #grabbing the last object of the array, should be the file name the 
                #user selected
                file_id = file_path_obj[length-1]
                
                imaging_analysis = ImagingAnalysis.ImagingAnalysis()
                li1 = "no preset min"
                li2 = "no preset max"
                #imaging_analysis.ec_coupling_signal_load(self.signal_coord)
                
                interp_selection = self.interp_drop.currentIndex() 
                analysis_type = self.analysis_drop.currentIndex()
                if analysis_type == 0: 
                    imaging_analysis.ec_coupling_map_act(li1, li2, transp, 
                                                         start_ind, end_ind,
                                                         interp_selection, file_id)
                if analysis_type == 1:   
                    apd_perc= float(self.perc_apd_edit_01.text())/100
                    imaging_analysis.ec_coupling_map_rep(self.data_fps, li1, 
                                                         li2, transp, start_ind,
                                                         end_ind, apd_perc,
                                                         interp_selection, file_id)
              


        # Generate data for succession of APDs
        if analysis_type == 2:
            # Grab the amplitude threshold, apd values and signals
            amp_thresh = float(self.max_val_edit.text())
            apd_input_01 = float(self.perc_apd_edit_01.text())/100
            apd_input_02 = float(self.perc_apd_edit_02.text())/100
            # Identify which signals have been selected for calculation
            ensemble_list = [self.ensemble_cb_01.isChecked(),
                             self.ensemble_cb_02.isChecked(),
                             self.ensemble_cb_03.isChecked(),
                             self.ensemble_cb_04.isChecked()]
            ind_analyze = self.signal_coord[ensemble_list, :]
            data_oap = []
            peak_ind = []
            peak_amp = []
            diast_ind = []
            act_ind = []
            apd_val_01 = []
            apd_val_02 = []
            apd_val_tri = []
            tau_fall = []
            f1_f0 = []
            d_f0 = []
            # Iterate through the code
            for idx in np.arange(len(ind_analyze)):
                data_oap.append(
                    self.data_filt[:, ind_analyze[idx][1],
                                   ind_analyze[idx][0]])
                # Calculate peak indices
                peak_ind.append(oap_peak_calc(data_oap[idx], start_ind,
                                              end_ind, amp_thresh,
                                              self.data_fps))
                # Calculate peak amplitudes
                peak_amp.append(data_oap[idx][peak_ind[idx]])
                # Calculate end-diastole indices
                diast_ind.append(diast_ind_calc(data_oap[idx],
                                                peak_ind[idx]))
                # Calculate the activation
                act_ind.append(act_ind_calc(data_oap[idx], diast_ind[idx],
                                            peak_ind[idx]))
                # Calculate the APD30
                apd_ind_01 = apd_ind_calc(data_oap[idx], end_ind,
                                          diast_ind[idx], peak_ind[idx],
                                          apd_input_01)
                apd_val_01.append(self.signal_time[apd_ind_01] -
                                  self.signal_time[act_ind[idx]])
                # Calculate APD80
                apd_ind_02 = apd_ind_calc(data_oap[idx], end_ind,
                                          diast_ind[idx], peak_ind[idx],
                                          apd_input_02)
                apd_val_02.append(self.signal_time[apd_ind_02] -
                                  self.signal_time[act_ind[idx]])
                # Calculate APD triangulation
                apd_val_tri.append(apd_val_02[idx]-apd_val_01[idx])
                # Calculate Tau Fall
                tau_fall.append(tau_calc(data_oap[idx], self.data_fps,
                                         peak_ind[idx], diast_ind[idx],
                                         end_ind))
                # Grab raw data, checking the data type to flip if necessary
                if self.image_type_drop.currentIndex() == 0:
                    # Membrane potential, flip the data
                    data = self.data*-1
                    
                elif self.image_type_drop.currentIndex() == 1:
                    # Calcium transient, don't flip the data
                    data = self.data
                        
                # Calculate the baseline fluorescence as the average of the
                # first 10 points
                f0 = np.average(data[:11,
                                     ind_analyze[idx][1],
                                     ind_analyze[idx][0]])
                # Calculate F1/F0 fluorescent ratio
                f1_f0.append(data[peak_ind[idx],
                                  ind_analyze[idx][1],
                                  ind_analyze[idx][0]]/f0)
                # Calculate D/F0 fluorescent ratio
                d_f0.append(data[diast_ind[idx],
                                 ind_analyze[idx][1],
                                 ind_analyze[idx][0]]/f0)
            # Open dialogue box for selecting the data directory
            save_fname = QFileDialog.getSaveFileName(
                self, "Save File", os.getcwd(), "Excel Files (*.xlsx)")
            # Write results to a spreadsheet
            ensemble_xlsx_print(save_fname[0], self.signal_time, ind_analyze,
                                data_oap, act_ind, peak_ind, tau_fall,
                                apd_input_01, apd_val_01, apd_input_02,
                                apd_val_02, apd_val_tri, d_f0, f1_f0,
                                self.image_type_drop.currentIndex())
            
        #Calculate and visualize Alternan 50% APD/CaD
        if analysis_type == 3:
            imaging_analysis = ImagingAnalysis.ImagingAnalysis()
            li1 = "no preset min"
            li2 = "no preset max"
            interp_selection = self.interp_drop.currentIndex()        
            start_time = float(self.start_time_edit.text())
            end_time = float(self.end_time_edit.text()) 
            
            #getting the file path the user selected
            file_path = str(self.file_path)
            #splitting the file path into an array by the /
            file_path_obj = file_path.split('/')
            #getting the length of this object
            length = len(file_path_obj)
            #grabbing the last object of the array, should be the file name the 
            #user selected
            file_id = file_path_obj[length-1]

            apd_input = float(self.perc_apd_edit_01.text())/100
            #call the alternan_50 function
            imaging_analysis.alternan_50(self.data_fps, self.data_filt, 
                                             li1, li2, transp, start_ind, 
                                             end_ind, start_ind2, end_ind2, 
                                             interp_selection, apd_input, 
                                             self.image_type_drop.currentIndex(),
                                             file_id)
            
        #Calculate and visualize moving Alternan
        if analysis_type == 4:
            imaging_analysis = ImagingAnalysis.ImagingAnalysis()
            li1 = "no preset min"
            li2 = "no preset max"
            interp_selection = self.interp_drop.currentIndex()        
            start_time = float(self.start_time_edit.text())
            end_time = float(self.end_time_edit.text())
            interp_selection = self.interp_drop.currentIndex()
            im_type=self.image_type_drop.currentIndex() 
            peak_coeff = float(self.perc_apd_edit_01.text())
            
            #getting the file path the user selected
            file_path = str(self.file_path)
            #splitting the file path into an array by the /
            file_path_obj = file_path.split('/')
            #getting the length of this object
            length = len(file_path_obj)
            #grabbing the last object of the array, should be the file name the 
            #user selected
            file_id = file_path_obj[length-1]
            
            imaging_analysis.moving_alternan(self.data_fps, self.data_filt, 
                                             li1, li2, transp, start_ind, 
                                             end_ind, im_type, 
                                             peak_coeff, file_id)
            
        #Calculate and visualize adjusted alternan
        if analysis_type == 5:
            imaging_analysis = ImagingAnalysis.ImagingAnalysis()
            li1 = "no preset min"
            li2 = "no preset max"
            interp_selection = self.interp_drop.currentIndex()        
            start_time = float(self.start_time_edit.text())
            end_time = float(self.end_time_edit.text())
            interp_selection = self.interp_drop.currentIndex() 
            peak_coeff = float(self.perc_apd_edit_01.text())
            
            #getting the file path the user selected
            file_path = str(self.file_path)
            #splitting the file path into an array by the /
            file_path_obj = file_path.split('/')
            #getting the length of this object
            length = len(file_path_obj)
            #grabbing the last object of the array, should be the file name the 
            #user selected
            file_id = file_path_obj[length-1]
            
            imaging_analysis.adjust_alternan(self.data_fps, self.data_filt, 
                                             li1, li2, transp, start_ind, 
                                             end_ind, interp_selection,
                                             peak_coeff, file_id)

        # S1-S2 map
        if analysis_type == 6:
            imaging_analysis = ImagingAnalysis.ImagingAnalysis()
            li1 = "no preset min"
            li2 = "no preset max"
            interp_selection = self.interp_drop.currentIndex()        
            start_time = float(self.start_time_edit.text())
            end_time = float(self.end_time_edit.text()) 
            peak_coeff = float(self.perc_apd_edit_02.text())
            apd_input = float(self.perc_apd_edit_01.text())/100
            threshold= float(self.max_val_edit.text())
            
            #getting the file path the user selected
            file_path = str(self.file_path)
            #splitting the file path into an array by the /
            file_path_obj = file_path.split('/')
            #getting the length of this object
            length = len(file_path_obj)
            #grabbing the last object of the array, should be the file name the 
            #user selected
            file_id = file_path_obj[length-1]
            
            imaging_analysis.S1_S2(self.data_fps, self.data_filt, 
                                             li1, li2, transp, start_ind, 
                                             end_ind, start_ind2, end_ind2, 
                                             interp_selection, apd_input, 
                                             self.image_type_drop.currentIndex(),
                                             peak_coeff, threshold, file_id)   
            
        # Calculate and visualize SNR
        if analysis_type == 7:
            # Calculate SNR
            self.snr = calc_snr(self.data_filt, start_ind, end_ind)
            # Grab the maximum SNR value
            max_val = np.nanmax(self.snr)
            # Generate a map of SNR
            self.snr_map = plt.figure()
            axes_snr_map = self.snr_map.add_axes([0.05, 0.1, 0.8, 0.8])
            transp = transp.astype(float)

            axes_snr_map.imshow(self.snr, alpha=transp, vmin=0,
                                vmax=max_val, cmap='jet')
            cax = plt.axes([0.87, 0.12, 0.05, 0.76])
            self.snr_map.colorbar(
                cm.ScalarMappable(
                    colors.Normalize(0, max_val),
                    cmap='jet'),
                cax=cax)
            
            #getting the file path the user selected
            file_path = str(self.file_path)
            #splitting the file path into an array by the /
            file_path_obj = file_path.split('/')
            #getting the length of this object
            length = len(file_path_obj)
            #grabbing the last object of the array, should be the file name the 
            #user selected
            file_id = file_path_obj[length-1]
            
            #making a folder if there isn't a "Saved Data Maps" folder
            if not os.path.exists("Saved Data Maps\\" + file_id):
               os.makedirs("Saved Data Maps\\" + file_id)
               
            #saving the data file
            savetxt('Saved Data Maps\\' + file_id + '\\snr.csv', self.snr, 
                    delimiter=',')
              
        # Repolarization map
        if analysis_type == 8: 
            # Calculate repolarization 
            self.act_ind = calc_tran_activation( 
                self.data_filt, start_ind, end_ind) 
            self.act_val = self.act_ind*(1/self.data_fps) 
            max_val = (end_ind-start_ind)*(1/self.data_fps) 
            # Grab the maximum APD value 
            final_apd = float(self.max_apd_edit.text()) 
            # Find the associated time index 
            max_apd_ind = abs(self.signal_time-(final_apd+start_time)) 
            max_apd_ind = np.argmin(max_apd_ind) 
            # Grab the percent APD 
            percent_apd = float(self.perc_apd_edit_01.text())/100 
            # Find the maximum amplitude of the action potential 
            max_amp_ind = np.argmax( 
                self.data_filt[ 
                    start_ind:max_apd_ind, :, :], axis=0 
                )+start_ind 
            # Preallocate variable for percent apd index and value 
            apd_ind = np.zeros(max_amp_ind.shape) 
            self.apd_val = np.zeros(max_amp_ind.shape) 
            # Step through the data 
            for n in np.arange(0, self.data_filt.shape[1]): 
                for m in np.arange(0, self.data_filt.shape[2]): 
                    # Ignore pixels that have been masked out 
                    if transp[n, m]: 
                        # Grab the data segment between max amp and end 
                        tmp = self.data_filt[ 
                            max_amp_ind[n, m]:max_apd_ind, n, m] 
                        # Find the minimum to find the index closest to 
                        # desired apd percent 
                        apd_ind[n, m] = np.argmin( 
                            abs( 
                                tmp-self.data_filt[max_amp_ind[n, m], n, m] * 
                                (1-percent_apd)) 
                            )+max_amp_ind[n, m]-start_ind 
                        # Subtract activation time to get apd 
                        self.apd_val[n, m] = (apd_ind[n, m] - 
                                              self.act_ind[n, m] 
                                              )*(1/self.data_fps) 
            # Generate a map of the action potential durations 
            self.apd_map = plt.figure() 
            axes_apd_map = self.apd_map.add_axes([0.05, 0.1, 0.8, 0.8]) 
            transp = transp.astype(float) 
            top = self.signal_time[max_apd_ind]-self.signal_time[start_ind] 
            max_apd_val = np.max(self.apd_val)
            axes_apd_map.imshow(self.apd_val, alpha=transp, vmin=0, 
                                vmax=max_apd_val, cmap='jet') 
            cax = plt.axes([0.87, 0.1, 0.05, 0.8]) 
            mm = self.apd_map.colorbar( 
                cm.ScalarMappable( 
                    colors.Normalize(0, top), cmap='jet'), 
                cax=cax, format='%.3f') 
             
            mm.mappable.set_clim(0,max_apd_val) 
            
            #getting the file path the user selected
            file_path = str(self.file_path)
            #splitting the file path into an array by the /
            file_path_obj = file_path.split('/')
            #getting the length of this object
            length = len(file_path_obj)
            #grabbing the last object of the array, should be the file name the 
            #user selected
            file_id = file_path_obj[length-1]
            
            #making a folder if there isn't a "Saved Data Maps" folder
            if not os.path.exists("Saved Data Maps\\" + file_id):
               os.makedirs("Saved Data Maps\\" + file_id)
               
            #saving the data file
            savetxt('Saved Data Maps\\' + file_id + '\\repolarization.csv', 
                    self.apd_val, delimiter=',')
            
    def map_analysis_cbar(self, progress_callback):
        #defining _translate
        #_translate = QtCore.QCoreApplication.translate
        li2 = float(self.Cbar_max.text())
        li1 = float(self.Cbar_min.text())
        # Grab analysis type
        analysis_type = self.analysis_drop.currentIndex()
        # Grab the start and end times
        start_time = float(self.start_time_edit.text())
        end_time = float(self.end_time_edit.text())
        
        if analysis_type == 3 or analysis_type == 6: 
            start_time2 = float(self.start_time_edit2.text())
            # Find the time index value to which the second top entry is closest
            start_ind2 = abs(self.signal_time-start_time2)
            start_ind2 = np.argmin(start_ind2)
            
            end_time2 = float(self.end_time_edit2.text())
            # Find the time index value to which the second top entry is closest
            end_ind2 = abs(self.signal_time-end_time2)
            end_ind2 = np.argmin(end_ind2)
            
        # Find the time index value to which the start entry is closest
        start_ind = abs(self.signal_time-start_time)
        start_ind = np.argmin(start_ind)
        # Find the time index value to which the top entry is closest
        end_ind = abs(self.signal_time-end_time)
        end_ind = np.argmin(end_ind)
        # Grab masking information
        transp = self.mask
        '''# Calculate activation
        self.act_ind = calc_tran_activation(
            self.data_filt, start_ind, end_ind)
        self.act_val = self.act_ind*(1/self.data_fps)
        max_val = (end_ind-start_ind)*(1/self.data_fps)'''
        
        imaging_analysis = ImagingAnalysis.ImagingAnalysis()
        
        
                # Calculate the APD 80
        if analysis_type == 1:
            if self.ec_coupling_cb.isChecked() == False:
                imaging_analysis = ImagingAnalysis.ImagingAnalysis()
                self.act_ind = calc_tran_activation(
                    self.data_filt, start_ind, end_ind)
                interp_selection = self.interp_drop.currentIndex()        
                start_time = float(self.start_time_edit.text())
                end_time = float(self.end_time_edit.text())
                apd_input = float(self.perc_apd_edit_01.text())/100
            
            #getting the file path the user selected
                file_path = str(self.file_path)
            #splitting the file path into an array by the /
                file_path_obj = file_path.split('/')
            #getting the length of this object
                length = len(file_path_obj)
            #grabbing the last object of the array, should be the file name the 
            #user selected
                file_id = file_path_obj[length-1]
                
                mapp2 = imaging_analysis.apd_analysis(self.data_fps, 
                                                      self.data_filt, start_ind, 
                                                      end_ind, interp_selection,
                                                      apd_input, file_id)[0]
            
                imaging_analysis.imaging_mapping(mapp2, li1, li2, transp)
        
        # Generate activation map
        if analysis_type == 0:
            if self.ec_coupling_cb.isChecked() == False:
                # Calculate activation
                self.act_ind = calc_tran_activation(
                    self.data_filt, start_ind, end_ind)
                self.act_val = self.act_ind*(1/self.data_fps)
                max_val = (end_ind-start_ind)*(1/self.data_fps)
                # Generate a map of the activation times
                self.act_map = plt.figure()
                axes_act_map = self.act_map.add_axes([0.05, 0.1, 0.8, 0.8])
                transp = transp.astype(float)
                
           
                axes_act_map.imshow(self.act_val, alpha=transp, vmin=li1,
                                    vmax=li2, cmap='jet')
    
                cax = plt.axes([0.87, 0.12, 0.05, 0.76])
                mm=self.act_map.colorbar(
                    cm.ScalarMappable(
                        colors.Normalize(0, max_val),
                        cmap='jet'),
                    cax=cax, format='%.3f')
                mm.mappable.set_clim(li1,li2)
                
                #getting the file path the user selected
                file_path = str(self.file_path)
                #splitting the file path into an array by the /
                file_path_obj = file_path.split('/')
                #getting the length of this object
                length = len(file_path_obj)
                #grabbing the last object of the array, should be the file name the 
                #user selected
                file_id = file_path_obj[length-1]
                
                #making a folder if there isn't a "Saved Data Maps" folder
                if not os.path.exists("Saved Data Maps\\" + file_id):
                   os.makedirs("Saved Data Maps\\" + file_id)
                   
                #saving the data file
                savetxt('Saved Data Maps\\' + file_id + '\\activation.csv', 
                        self.act_val, delimiter=',')
                
        if self.ec_coupling_cb.isChecked() == True: 
                #getting the file path the user selected
                file_path = str(self.file_path)
                #splitting the file path into an array by the /
                file_path_obj = file_path.split('/')
                #getting the length of this object
                length = len(file_path_obj)
                #grabbing the last object of the array, should be the file name the 
                #user selected
                file_id = file_path_obj[length-1]
                
                imaging_analysis = ImagingAnalysis.ImagingAnalysis()
          
                #imaging_analysis.ec_coupling_signal_load(self.signal_coord)
                
                interp_selection = self.interp_drop.currentIndex() 
                analysis_type = self.analysis_drop.currentIndex()
                if analysis_type == 0: 
                    imaging_analysis.ec_coupling_map_act(li1, li2, transp, 
                                                         start_ind, end_ind,
                                                         interp_selection, file_id)
                if analysis_type == 1:   
                    apd_perc= float(self.perc_apd_edit_01.text())/100
                    imaging_analysis.ec_coupling_map_rep(self.data_fps, li1, 
                                                         li2, transp, start_ind,
                                                         end_ind, apd_perc,
                                                         interp_selection, file_id)

            
        # Generate data for succession of APDs
        if analysis_type == 2:
            # Grab the amplitude threshold, apd values and signals
            amp_thresh = float(self.max_val_edit.text())
            apd_input_01 = float(self.perc_apd_edit_01.text())/100
            apd_input_02 = float(self.perc_apd_edit_02.text())/100
            # Identify which signals have been selected for calculation
            ensemble_list = [self.ensemble_cb_01.isChecked(),
                             self.ensemble_cb_02.isChecked(),
                             self.ensemble_cb_03.isChecked(),
                             self.ensemble_cb_04.isChecked()]
            ind_analyze = self.signal_coord[ensemble_list, :]
            data_oap = []
            peak_ind = []
            peak_amp = []
            diast_ind = []
            act_ind = []
            apd_val_01 = []
            apd_val_02 = []
            apd_val_tri = []
            tau_fall = []
            f1_f0 = []
            d_f0 = []
            # Iterate through the code
            for idx in np.arange(len(ind_analyze)):
                data_oap.append(
                    self.data_filt[:, ind_analyze[idx][1],
                                   ind_analyze[idx][0]])
                # Calculate peak indices
                peak_ind.append(oap_peak_calc(data_oap[idx], start_ind,
                                              end_ind, amp_thresh,
                                              self.data_fps))
                # Calculate peak amplitudes
                peak_amp.append(data_oap[idx][peak_ind[idx]])
                # Calculate end-diastole indices
                diast_ind.append(diast_ind_calc(data_oap[idx],
                                                peak_ind[idx]))
                # Calculate the activation
                act_ind.append(act_ind_calc(data_oap[idx], diast_ind[idx],
                                            peak_ind[idx]))
                # Calculate the APD30
                apd_ind_01 = apd_ind_calc(data_oap[idx], end_ind,
                                          diast_ind[idx], peak_ind[idx],
                                          apd_input_01)
                apd_val_01.append(self.signal_time[apd_ind_01] -
                                  self.signal_time[act_ind[idx]])
                # Calculate APD80
                apd_ind_02 = apd_ind_calc(data_oap[idx], end_ind,
                                          diast_ind[idx], peak_ind[idx],
                                          apd_input_02)
                apd_val_02.append(self.signal_time[apd_ind_02] -
                                  self.signal_time[act_ind[idx]])
                # Calculate APD triangulation
                apd_val_tri.append(apd_val_02[idx]-apd_val_01[idx])
                # Calculate Tau Fall
                tau_fall.append(tau_calc(data_oap[idx], self.data_fps,
                                         peak_ind[idx], diast_ind[idx],
                                         end_ind))
                # Grab raw data, checking the data type to flip if necessary
                if self.image_type_drop.currentIndex() == 0:
                    # Membrane potential, flip the data
                    data = self.data*-1
                        
                elif self.image_type_drop.currentIndex() == 1:
                    # Calcium transient, don't flip the data
                    data = self.data

                # Calculate the baseline fluorescence as the average of the
                # first 10 points
                f0 = np.average(data[:11,
                                     ind_analyze[idx][1],
                                     ind_analyze[idx][0]])
                # Calculate F1/F0 fluorescent ratio
                f1_f0.append(data[peak_ind[idx],
                                  ind_analyze[idx][1],
                                  ind_analyze[idx][0]]/f0)
                # Calculate D/F0 fluorescent ratio
                d_f0.append(data[diast_ind[idx],
                                 ind_analyze[idx][1],
                                 ind_analyze[idx][0]]/f0)
            # Open dialogue box for selecting the data directory
            save_fname = QFileDialog.getSaveFileName(
                self, "Save File", os.getcwd(), "Excel Files (*.xlsx)")
            # Write results to a spreadsheet
            ensemble_xlsx_print(save_fname[0], self.signal_time, ind_analyze,
                                data_oap, act_ind, peak_ind, tau_fall,
                                apd_input_01, apd_val_01, apd_input_02,
                                apd_val_02, apd_val_tri, d_f0, f1_f0,
                                self.image_type_drop.currentIndex())
            
        #Calculate and visualize Alternan 50% APD/CaD
        if analysis_type == 3:
            imaging_analysis = ImagingAnalysis.ImagingAnalysis()
            interp_selection = self.interp_drop.currentIndex()        
            start_time = float(self.start_time_edit.text())
            end_time = float(self.end_time_edit.text())
            interp_selection = self.interp_drop.currentIndex()  
            apd_input = float(self.perc_apd_edit_01.text())/100
            
            #getting the file path the user selected
            file_path = str(self.file_path)
            #splitting the file path into an array by the /
            file_path_obj = file_path.split('/')
            #getting the length of this object
            length = len(file_path_obj)
            #grabbing the last object of the array, should be the file name the 
            #user selected
            file_id = file_path_obj[length-1]
                
            #call the alternan_50 function
            imaging_analysis.alternan_50(self.data_fps, self.data_filt, 
                                             li1, li2, transp, start_ind, 
                                             end_ind, start_ind2, end_ind2, 
                                             interp_selection, apd_input,
                                             self.image_type_drop.currentIndex(),
                                             file_id)
            
        #Calculate and visualize moving Alternan
        if analysis_type == 4:
            imaging_analysis = ImagingAnalysis.ImagingAnalysis()
            interp_selection = self.interp_drop.currentIndex()        
            start_time = float(self.start_time_edit.text())
            end_time = float(self.end_time_edit.text())
            interp_selection = self.interp_drop.currentIndex()  
            im_type=self.image_type_drop.currentIndex()       
            peak_coeff = float(self.perc_apd_edit_01.text())
            
            #getting the file path the user selected
            file_path = str(self.file_path)
            #splitting the file path into an array by the /
            file_path_obj = file_path.split('/')
            #getting the length of this object
            length = len(file_path_obj)
            #grabbing the last object of the array, should be the file name the 
            #user selected
            file_id = file_path_obj[length-1]
                
            imaging_analysis.moving_alternan(self.data_fps, self.data_filt, 
                                             li1, li2, transp, start_ind, 
                                             end_ind, im_type, 
                                             peak_coeff, file_id)
            
        #Calculate and visualize adjusted alternan
        if analysis_type == 5:
            imaging_analysis = ImagingAnalysis.ImagingAnalysis()
            interp_selection = self.interp_drop.currentIndex()        
            start_time = float(self.start_time_edit.text())
            end_time = float(self.end_time_edit.text())
            interp_selection = self.interp_drop.currentIndex()  
            peak_coeff = float(self.perc_apd_edit_01.text())
            
            #getting the file path the user selected
            file_path = str(self.file_path)
            #splitting the file path into an array by the /
            file_path_obj = file_path.split('/')
            #getting the length of this object
            length = len(file_path_obj)
            #grabbing the last object of the array, should be the file name the 
            #user selected
            file_id = file_path_obj[length-1]
                
            imaging_analysis.adjust_alternan(self.data_fps, self.data_filt, 
                                             li1, li2, transp, start_ind, 
                                             end_ind, interp_selection,
                                             peak_coeff, file_id)   
            
        # S1 S2 map
        if analysis_type == 6:
            imaging_analysis = ImagingAnalysis.ImagingAnalysis()
            interp_selection = self.interp_drop.currentIndex()        
            start_time = float(self.start_time_edit.text())
            end_time = float(self.end_time_edit.text()) 
            peak_coeff = float(self.perc_apd_edit_02.text())
            apd_input = float(self.perc_apd_edit_01.text())/100
            threshold = float(self.max_val_edit.text())
            
            #getting the file path the user selected
            file_path = str(self.file_path)
            #splitting the file path into an array by the /
            file_path_obj = file_path.split('/')
            #getting the length of this object
            length = len(file_path_obj)
            #grabbing the last object of the array, should be the file name the 
            #user selected
            file_id = file_path_obj[length-1]
                
            imaging_analysis.S1_S2(self.data_fps, self.data_filt, 
                                             li1, li2, transp, start_ind, 
                                             end_ind, start_ind2, end_ind2, 
                                             interp_selection, apd_input, 
                                             self.image_type_drop.currentIndex(),
                                             peak_coeff, threshold, file_id)       
        # Calculate and visualize SNR
        if analysis_type == 7:
            # Calculate SNR
            self.snr = calc_snr(self.data_filt, start_ind, end_ind)
            # Grab the maximum SNR value
            max_val = np.nanmax(self.snr)
            # Generate a map of SNR
            self.snr_map = plt.figure()
            axes_snr_map = self.snr_map.add_axes([0.05, 0.1, 0.8, 0.8])
            transp = transp.astype(float)
            axes_snr_map.imshow(self.snr, alpha=transp, vmin=li1,
                                vmax=li2, cmap='jet')
            cax = plt.axes([0.87, 0.12, 0.05, 0.76])
            mm=self.snr_map.colorbar(
                cm.ScalarMappable(
                    colors.Normalize(0, max_val),
                    cmap='jet'),
                cax=cax)
            mm.mappable.set_clim(li1,li2)
            
            #getting the file path the user selected
            file_path = str(self.file_path)
            #splitting the file path into an array by the /
            file_path_obj = file_path.split('/')
            #getting the length of this object
            length = len(file_path_obj)
            #grabbing the last object of the array, should be the file name the 
            #user selected
            file_id = file_path_obj[length-1]
            
            #making a folder if there isn't a "Saved Data Maps" folder
            if not os.path.exists("Saved Data Maps\\" + file_id):
               os.makedirs("Saved Data Maps\\" + file_id)
               
            #saving the data file
            savetxt('Saved Data Maps\\' + file_id + '\\snr.csv', self.snr, 
                    delimiter=',')
            
        # Repolarization Map
        if analysis_type == 8: 
            # Calculate repolarization 
            self.act_ind = calc_tran_activation( 
                self.data_filt, start_ind, end_ind) 
            self.act_val = self.act_ind*(1/self.data_fps) 
            max_val = (end_ind-start_ind)*(1/self.data_fps) 
            # Grab the maximum APD value 
            final_apd = float(self.max_apd_edit.text()) 
            # Find the associated time index 
            max_apd_ind = abs(self.signal_time-(final_apd+start_time)) 
            max_apd_ind = np.argmin(max_apd_ind) 
            # Grab the percent APD 
            percent_apd = float(self.perc_apd_edit_01.text())/100 
            # Find the maximum amplitude of the action potential 
            max_amp_ind = np.argmax( 
                self.data_filt[ 
                    start_ind:max_apd_ind, :, :], axis=0 
                )+start_ind 
            # Preallocate variable for percent apd index and value 
            apd_ind = np.zeros(max_amp_ind.shape) 
            self.apd_val = np.zeros(max_amp_ind.shape) 
            # Step through the data 
            for n in np.arange(0, self.data_filt.shape[1]): 
                for m in np.arange(0, self.data_filt.shape[2]): 
                    # Ignore pixels that have been masked out 
                    if transp[n, m]: 
                        # Grab the data segment between max amp and end 
                        tmp = self.data_filt[ 
                            max_amp_ind[n, m]:max_apd_ind, n, m] 
                        # Find the minimum to find the index closest to 
                        # desired apd percent 
                        apd_ind[n, m] = np.argmin( 
                            abs( 
                                tmp-self.data_filt[max_amp_ind[n, m], n, m] * 
                                (1-percent_apd)) 
                            )+max_amp_ind[n, m]-start_ind 
                        # Subtract activation time to get apd 
                        self.apd_val[n, m] = (apd_ind[n, m] - 
                                              self.act_ind[n, m] 
                                              )*(1/self.data_fps) 
            # Generate a map of the action potential durations 
            self.apd_map = plt.figure() 
            axes_apd_map = self.apd_map.add_axes([0.05, 0.1, 0.8, 0.8]) 
            transp = transp.astype(float) 
            top = self.signal_time[max_apd_ind]-self.signal_time[start_ind] 
            axes_apd_map.imshow(self.apd_val, alpha=transp, vmin=li1, 
                                vmax=li2, cmap='jet') 
            cax = plt.axes([0.87, 0.1, 0.05, 0.8]) 
            mm = self.apd_map.colorbar( 
                cm.ScalarMappable( 
                    colors.Normalize(0, top), cmap='jet'), 
                cax=cax, format='%.3f') 
            mm.mappable.set_clim(li1,li2)
            
            #getting the file path the user selected
            file_path = str(self.file_path)
            #splitting the file path into an array by the /
            file_path_obj = file_path.split('/')
            #getting the length of this object
            length = len(file_path_obj)
            #grabbing the last object of the array, should be the file name the 
            #user selected
            file_id = file_path_obj[length-1]
            
            #making a folder if there isn't a "Saved Data Maps" folder
            if not os.path.exists("Saved Data Maps\\" + file_id):
               os.makedirs("Saved Data Maps\\" + file_id)
               
            #saving the data file
            savetxt('Saved Data Maps\\' + file_id + '\\repolarization.csv', 
                    self.apd_val, delimiter=',')
            
    #creating a function to organize the post data analysis and only show the means
    def mean_post_mapping_data_analysis(self):
        #getting the file path the user selected
        file_path = str(self.file_path)
        #splitting the file path into an array by the /
        file_path_obj = file_path.split('/')
        #getting the length of this object
        length = len(file_path_obj)
        #grabbing the last object of the array, should be the file name the 
        #user selected
        file_id = file_path_obj[length-1]
        #calling the mean post analysis function
        ImagingAnalysis.ImagingAnalysis().mean_post_analysis(file_id)
    
    #creating a function to analyze the data with regional analysis
    def post_mapping_data_analysis(self):
        #setting the mean post mapping analysis button to true
        self.mean_post_mapping_analysis.setEnabled(True)
        li1 = 0
        li2 = 70
        
        #getting the file path the user selected
        file_path = str(self.file_path)
        #splitting the file path into an array by the /
        file_path_obj = file_path.split('/')
        #getting the length of this object
        length = len(file_path_obj)
        #grabbing the last object of the array, should be the file name the 
        #user selected
        file_id = file_path_obj[length-1]
        
        #calling the post analysis function
        ImagingAnalysis.ImagingAnalysis().post_analysis(li1, li2, file_id, self.file_name)
        
    def export_data_numeric(self):
        # Determine if data is prepped or unprepped
        if self.preparation_tracker == 0:
            data = self.data_prop
        else:
            data = self.data_filt
        # Grab oaps
        data_oap = []
        for idx in np.arange(0, 4):
            if self.signal_toggle[idx] == 1:
                data_oap.append(
                    data[:, self.signal_coord[idx, 1],
                         self.signal_coord[idx, 0]])
        # Open dialogue box for selecting the data directory
        save_fname = QFileDialog.getSaveFileName(
            self, "Save File", os.getcwd(), "Excel Files (*.xlsx)")
        # Write results to a spreadsheet
        signal_data_xlsx_print(save_fname[0], self.signal_time, data_oap,
                               self.signal_coord, self.data_fps)

    def signal_select(self):
        # Create a button press event
        self.cid = self.mpl_canvas.mpl_connect(
            'button_press_event', self.on_click)
        self.reset_signal_button.setEnabled(True)

    def signal_select_edit(self):
        if self.signal_emit_done == 1:
            # Update the tracker to negative (i.e., 0) and continue
            self.signal_emit_done = 0
            # Grab all of the values and make sure they are integer values
            for n in np.arange(4):
                # Create iteration names for the x and y structures
                xname = 'sig{}_x_edit'.format(n+1)
                x = getattr(self, xname)
                yname = 'sig{}_y_edit'.format(n+1)
                y = getattr(self, yname)
                # Check to see if there is an empty edit box in the pair
                if x.text() == '' or y.text() == '':
                    continue
                else:
                    # Make sure the entered values are numeric
                    try:
                        new_x = int(x.text())
                    except ValueError:
                        self.sig_win_warn(3)
                        x.setText(str(self.signal_coord[n][0]))
                        self.signal_emit_done = 1
                        break
                    try:
                        new_y = int(y.text())
                    except ValueError:
                        self.sig_win_warn(3)
                        y.setText(str(self.signal_coord[n][1]))
                        self.signal_emit_done = 1
                        break
                    # Grab the current string values and convert to integers
                    coord_ints = [new_x, new_y]
                    # Check to make sure the coordinates are within range
                    if coord_ints[0] < 0 or (
                            coord_ints[0] >= self.data.shape[2]):
                        self.sig_win_warn(2)
                        x.setText(str(self.signal_coord[n][0]))
                        self.signal_emit_done = 1
                        break
                    elif coord_ints[1] < 0 or (
                            coord_ints[1] >= self.data.shape[1]):
                        self.sig_win_warn(2)
                        y.setText(str(self.signal_coord[n][1]))
                        self.signal_emit_done = 1
                        break
                    # Place integer values in global signal coordinate variable
                    self.signal_coord[n] = coord_ints
                    # Convert integers to strings and update the edit boxes
                    x.setText(str(coord_ints[0]))
                    y.setText(str(coord_ints[1]))
                    # Make sure the axes is toggled on for plotting
                    self.signal_toggle[n] = 1
                    # Make sure the APD Ensemble check box is enabled
                    cb_name = 'ensemble_cb_0{}'.format(n+1)
                    cb = getattr(self, cb_name)
                    cb.setChecked(True)
                    # Check to see if the next edit boxes should be toggled on
                    if sum(self.signal_toggle) < 4:
                        # Grab the number of active axes
                        act_ax = int(sum(self.signal_toggle))
                        # Activate the next set of edit boxes
                        xname = 'sig{}_x_edit'.format(act_ax+1)
                        x = getattr(self, xname)
                        x.setEnabled(True)
                        yname = 'sig{}_y_edit'.format(act_ax+1)
                        y = getattr(self, yname)
                        y.setEnabled(True)
                        # Update the select signal button index
                        self.signal_ind = int(sum(self.signal_toggle))
            # Update the axes
            self.update_axes()
            self.signal_emit_done = 1

    def update_win(self):
        bot_val = float(self.axes_start_time_edit.text())
        top_val = float(self.axes_end_time_edit.text())
        # Find the time index value to which the bot entry is closest
        bot_ind = abs(self.signal_time-bot_val)
        self.axes_start_ind = np.argmin(bot_ind)
        # Adjust the start time string accordingly
        self.axes_start_time_edit.setText(
            str(self.signal_time[self.axes_start_ind]))
        # Find the time index value to which the top entry is closest
        top_ind = abs(self.signal_time-top_val)
        self.axes_end_ind = np.argmin(top_ind)
        # Adjust the end time string accordingly
        self.axes_end_time_edit.setText(
            str(self.signal_time[self.axes_end_ind]))

        # Update the signal axes
        self.update_axes()

    def update_analysis_win(self):
        # Grab new start time value index and update entry to actual value
        if self.start_time_edit.text():
            bot_val = float(self.start_time_edit.text())
            # Find the time index value to which the bot entry is closest
            bot_ind = abs(self.signal_time-bot_val)
            self.anal_start_ind = np.argmin(bot_ind)
            # Adjust the start time string accordingly
            self.start_time_edit.setText(
                str(self.signal_time[self.anal_start_ind]))
            # Set boolean to true to signal axes updates accordingly
            self.analysis_bot_lim = True
        else:
            # Set the start time variable to empty
            self.anal_start_ind = []
            # Set boolean to false so it no longer updates
            self.analysis_bot_lim = False
        # Grab new end time value index and update entry to actual value
        if self.end_time_edit.text():
            top_val = float(self.end_time_edit.text())
            # Find the time index value to which the top entry is closest
            top_ind = abs(self.signal_time-top_val)
            self.anal_end_ind = np.argmin(top_ind)
            # Adjust the end time string accordingly
            self.end_time_edit.setText(
                str(self.signal_time[self.anal_end_ind]))
            # Set boolean to true to signal axes updates accordingly
            self.analysis_top_lim = True
        else:
            # Set the start time variable to empty
            self.anal_end_ind = []
            # Set boolean to false so it no longer updates
            self.analysis_top_lim = False
        # Grab new end time value index and update entry to actual value
        if self.start_time_edit2.text():
            bot_val2 = float(self.start_time_edit2.text())
            # Find the time index value to which the top entry is closest
            bot_ind2 = abs(self.signal_time-bot_val2)
            self.anal_start_ind2 = np.argmin(bot_ind2)
            # Adjust the end time string accordingly
            self.start_time_edit2.setText(
                str(self.signal_time[self.anal_start_ind2]))
            # Set boolean to true to signal axes updates accordingly
            self.analysis_bot_lim2 = True
        else:
            # Set the end time variable to empty
            self.anal_start_ind2 = []
            # Set boolean to false so it no longer updates
            self.analysis_bot_lim2 = False

        if self.end_time_edit2.text():
            top_val2 = float(self.end_time_edit2.text())
            # Find the time index value to which the top entry is closest
            top_ind2 = abs(self.signal_time-top_val2)
            self.anal_end_ind2 = np.argmin(top_ind2)
            # Adjust the end time string accordingly
            self.end_time_edit2.setText(
                str(self.signal_time[self.anal_end_ind2]))
            # Set boolean to true to signal axes updates accordingly
            self.analysis_top_lim2 = True
        else:
            # Set the end time variable to empty
            self.anal_end_ind2 = []
            # Set boolean to false so it no longer updates
            self.analysis_top_lim2 = False

        # Grab new max amplitude value and update entry to actual value
        if self.max_val_edit.text():
            # Set boolean to true to signal axes updates accordingly
            self.analysis_y_lim = True
        else:
            # Set boolean to false so it no longer updates
            self.analysis_y_lim = False
        # Update the axes accordingly
        self.update_axes()

    def play_movie(self):
        # Grab the current value of the movie scroll bar
        cur_val = self.movie_scroll_obj.value()
        # Grab the maximum value of the movie scroll bar
        max_val = self.movie_scroll_obj.maximum()
        # Pass the function to execute
        self.runner = JobRunner(self.update_frame, (cur_val, max_val))
        self.runner.signals.progress.connect(self.movie_progress)
        # Set or reset the pause variable and activate the pause button
        self.is_paused = False
        self.pause_button.setEnabled(True)
        # Execute
        self.threadpool.start(self.runner)

    def update_frame(self, vals, progress_callback):
        # Start at the current frame and proceed to the end of the file
        for n in np.arange(vals[0]+5, vals[1], 5):
            # Create a minor delay so change is noticeable
            time.sleep(0.5)
            # Emit a signal that will trigger the movie_progress function
            progress_callback.emit(n)
            # If the pause button is hit, break the loop
            if self.is_paused:
                self.pause_button.setEnabled(False)
                break
        # At the end deactivate the pause button
        self.pause_button.setEnabled(False)

    def movie_progress(self, n):
        # Update the scroll bar value, thereby updating the movie screen
        self.movie_scroll_obj.setValue(n)
        # Return the movie screen
        return self.mpl_canvas.fig

    # Function for pausing the movie once the play button has been hit
    def pause_movie(self):
        self.is_paused = True

    # Export movie of ovelayed optical data
    def export_movie(self):
        # Open dialogue box for selecting the file name
        save_fname = QFileDialog.getSaveFileName(
            self, "Save File", os.getcwd(), "mp4 Files (*.mp4)")
        # The function for grabbing the video frames
        animation = FuncAnimation(self.mpl_canvas.fig, self.movie_progress,
                                  np.arange(0, self.data.shape[0], 5),
                                  fargs=[], interval=self.data_fps)
        # Execute the function
        animation.save(save_fname[0],
                       dpi=self.mpl_canvas.fig.dpi)

    # Rotate image 90 degrees counterclockwise function
    def rotate_image_ccw90(self):
        # Rotate the data 90 degress counterclockwise
        self.data = np.rot90(self.data, k=1, axes=(1, 2))
        #removing the first 5 tiffs to get rid of the blip from lights 
        #being turned on
        n = 5
        self.data = self.data[n:]
        # Rotate the bacground image 90 degrees counterclockwise
        self.im_bkgd = np.rot90(self.im_bkgd, k=1)
        # Update variable for tracking rotation
        if self.rotate_tracker < 3:
            self.rotate_tracker += 1
        else:
            self.rotate_tracker = 0
        # Swap crop box values
        self.crop_bound_rot()
        # Update cropping strings according to new image dimensions
        self.crop_update()

    # Rotate image 90 degress clockwise function
    def rotate_image_cw90(self):
        # Rotate the data 90 degress clockwise
        self.data = np.rot90(self.data, k=-1, axes=(1, 2))
        #removing the first 5 tiffs to get rid of the blip from lights 
        #being turned on
        n = 5
        self.data = self.data[n:]
        # Rotate the bacground image 90 degrees clockwise
        self.im_bkgd = np.rot90(self.im_bkgd, k=-1)
        # Update variable for tracking rotation
        if self.rotate_tracker == 0:
            self.rotate_tracker = 3
        else:
            self.rotate_tracker -= 1
        # Swap crop box values
        self.crop_bound_rot()
        # Update cropping strings according to new image dimensions
        self.crop_update()

    # Rotate cropping bounding box
    def crop_bound_rot(self):
        # Grab the current values
        new_x = [int(self.crop_ylower_edit.text()),
                 int(self.crop_yupper_edit.text())]
        new_y = [int(self.crop_xlower_edit.text()),
                 int(self.crop_xupper_edit.text())]
        # Replace x values
        self.crop_xlower_edit.setText(str(new_x[0]))
        self.crop_xupper_edit.setText(str(new_x[1]))
        # Replace y values
        self.crop_ylower_edit.setText(str(new_y[0]))
        self.crop_yupper_edit.setText(str(new_y[1]))

    # Enable the crop limit boxes
    def crop_enable(self):
        if self.crop_cb.isChecked():
            # Enable the labels and edit boxes for cropping
            self.crop_xlabel.setEnabled(True)
            self.crop_xlower_edit.setEnabled(True)
            self.crop_xupper_edit.setEnabled(True)
            self.crop_ylabel.setEnabled(True)
            self.crop_ylower_edit.setEnabled(True)
            self.crop_yupper_edit.setEnabled(True)
            # Update the axes
            self.update_axes()
        else:
            # Disable the labesl and edit boxes for cropping
            self.crop_xlabel.setEnabled(False)
            self.crop_xlower_edit.setEnabled(False)
            self.crop_xupper_edit.setEnabled(False)
            self.crop_ylabel.setEnabled(False)
            self.crop_ylower_edit.setEnabled(False)
            self.crop_yupper_edit.setEnabled(False)

    def crop_update(self):
        if self.signal_emit_done == 1:
            # Create variable for stopping double tap
            self.signal_emit_done = 0
            # Check to make sure the x coordinates are within the image bounds
            try:
                new_x = [int(self.crop_xlower_edit.text()),
                         int(self.crop_xupper_edit.text())]
            except ValueError:
                self.sig_win_warn(3)
                self.crop_xlower_edit.setText(str(self.crop_xbound[0]))
                self.crop_xupper_edit.setText(str(self.crop_xbound[1]))
            else:
                # Update the bounds of the crop box
                if (new_x[0] < 0 or new_x[0] > self.data.shape[2] or
                        new_x[1] < 0 or new_x[1] > self.data.shape[2]):
                    self.sig_win_warn(2)
                    self.crop_xlower_edit.setText(str(self.crop_xbound[0]))
                    self.crop_xupper_edit.setText(str(self.crop_xbound[1]))
                elif new_x[0] >= new_x[1]:
                    self.sig_win_warn(4)
                    self.crop_xlower_edit.setText(str(self.crop_xbound[0]))
                    self.crop_xupper_edit.setText(str(self.crop_xbound[1]))
                else:
                    self.crop_xbound = [new_x[0], new_x[1]]
            # Check to make sure the y coordinates are within the image bounds
            try:
                new_y = [int(self.crop_ylower_edit.text()),
                         int(self.crop_yupper_edit.text())]
            except ValueError:
                self.sig_win_warn(3)
                self.crop_ylower_edit.setText(str(self.crop_ybound[0]))
                self.crop_yupper_edit.setText(str(self.crop_ybound[1]))
            else:
                # Update the bounds of the crop box
                if (new_y[0] < 0 or new_y[0] > self.data.shape[1] or
                        new_y[1] < 0 or new_y[1] > self.data.shape[1]):
                    self.sig_win_warn(2)
                    self.crop_ylower_edit.setText(str(self.crop_ybound[0]))
                    self.crop_yupper_edit.setText(str(self.crop_ybound[1]))
                elif new_y[0] >= new_y[1]:
                    self.sig_win_warn(4)
                    self.crop_ylower_edit.setText(str(self.crop_ybound[0]))
                    self.crop_yupper_edit.setText(str(self.crop_ybound[1]))
                else:
                    self.crop_ybound = [new_y[0], new_y[1]]
            # Update the image axis
            self.update_axes()
            # Indicate function has ended
            self.signal_emit_done = 1

    # ASSIST (I.E., NON-BUTTON) FUNCTIONS
    # Function for grabbing the x and y coordinates of a button click
    def on_click(self, event):
        # Grab the axis coordinates of the click event
        self.signal_coord[self.signal_ind] = [round(event.xdata),
                                              round(event.ydata)]
        self.signal_coord = self.signal_coord.astype(int)
        # Update the toggle variable to indicate points should be plotted
        self.signal_toggle[self.signal_ind] = 1
        # Update the plots accordingly
        self.update_axes()
        # Check the associated check box
        checkboxname = 'ensemble_cb_0{}'.format(self.signal_ind+1)
        checkbox = getattr(self, checkboxname)
        checkbox.setChecked(True)
        # Populate the signal coordinate edit boxes
        sigx_name = 'sig{}_x_edit'.format(self.signal_ind+1)
        sigx = getattr(self, sigx_name)
        sigx.setText(str(self.signal_coord[self.signal_ind][0]))
        sigy_name = 'sig{}_y_edit'.format(self.signal_ind+1)
        sigy = getattr(self, sigy_name)
        sigy.setText(str(self.signal_coord[self.signal_ind][1]))
        # Check to see if the next edit boxes should be toggled on
        if sum(self.signal_toggle) < 4:
            # Grab the number of active axes
            act_ax = int(sum(self.signal_toggle))
            # Activate the next set of edit boxes
            xname = 'sig{}_x_edit'.format(act_ax+1)
            x = getattr(self, xname)
            x.setEnabled(True)
            yname = 'sig{}_y_edit'.format(act_ax+1)
            y = getattr(self, yname)
            y.setEnabled(True)
        # Update the index of the signal for next selection
        if self.signal_ind == 3:
            self.signal_ind = 0
        else:
            self.signal_ind += 1
        # End the button press event
        self.mpl_canvas.mpl_disconnect(self.cid)

    # Function for grabbing the x and y coordinates of button clicks for an
    # add remove polygon
    # function is unused
    def poly_click(self, event):
        print(f'Starting coordinates: {self.poly_coord}')
        # while self.poly_coord.shape[0] < 5:
        # Grab the axis coordinates of the click event
        if self.poly_start:
            # Add new button click to the array
            self.poly_coord = np.vstack(
                (self.poly_coord,
                  [[round(event.xdata), round(event.ydata)]]))
        else:
            # Create a new button click array
            self.poly_coord = np.array(
                [round(event.xdata), round(event.ydata)])
            # Set poly_start to True
            self.poly_start = True
        print(f'Final coordinates: {self.poly_coord}')
        # End the button press event
        self.mpl_canvas.mpl_disconnect(self.pid)
        # End holding pattern
        self.poly_running = False

    # Function for entering out-of-range values for signal window view
    def sig_win_warn(self, ind):
        # Create a message box to communicate the absence of data
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        if ind == 0:
            msg.setText(
                "Entry must be a numeric value between 0 and {.2f}!".format(
                    self.signal_time[-1]))
        elif ind == 1:
            msg.setText("The Start Time must be less than the End Time!")
        elif ind == 2:
            msg.setText("Entered coordinates outside image dimensions!")
        elif ind == 3:
            msg.setText("Entered value must be numeric!")
        elif ind == 4:
            msg.setText("Lower limit must be less than upper limit!")
        msg.setWindowTitle("Warning")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    # Function for updating the axes
    def update_axes(self):
        # UPDATE THE IMAGE AXIS
        # Determine if data is prepped or unprepped
        if self.preparation_tracker == 0:
            data = self.data_prop
        else:
            data = self.data_filt
        # UPDATE THE OPTICAL IMAGE AXIS
        # Clear axis for update
        self.mpl_canvas.axes.cla()
        # Update the UI with an image off the top of the stack
        self.mpl_canvas.axes.imshow(self.im_bkgd, cmap='gray')
        # Match the matplotlib figure background color to the GUI
        self.mpl_canvas.fig.patch.set_facecolor(self.bkgd_color)
        # If normalized, overlay the potential values
        if self.norm_flag == 1:
            # Get the current value of the movie slider
            sig_id = self.movie_scroll_obj.value()
            # Create the transparency mask
            #mask = self.mask
            self.mpl_canvas.axes.imshow(self.data_filt[sig_id, :, :],
                                        alpha=0.5, vmin=0, vmax=1,
                                        cmap='jet')
        # Check to see if signals have been selected and activate export tools
        if self.signal_ind != 1:
            self.export_data_button.setEnabled(True)
        # Plot the select signal points
        for cnt, ind in enumerate(self.signal_coord):
            if self.signal_toggle[cnt] == 0:
                continue
            else:
                self.mpl_canvas.axes.scatter(
                    ind[0], ind[1], color=self.cnames[cnt])
        # Check to see if polygon plotting is turned on
        if self.poly_toggle:
            self.mpl_canvas.axes.plot(self.poly_coord[:, 0],
                                      self.poly_coord[:, 1], color='#0000ff')

        # Check to see if crop is being utilized
        if self.crop_cb.isChecked():
            if self.data_prop_button.text() == 'Save Properties':
                # Plot vertical sides of bounding box
                self.mpl_canvas.axes.plot(
                    [self.crop_xbound[0], self.crop_xbound[0]],
                    [self.crop_ybound[0], self.crop_ybound[1]],
                    color='orange')
                self.mpl_canvas.axes.plot(
                    [self.crop_xbound[1], self.crop_xbound[1]],
                    [self.crop_ybound[0], self.crop_ybound[1]],
                    color='orange')
                # Plot horizontal sides of bounding box
                self.mpl_canvas.axes.plot(
                    [self.crop_xbound[0], self.crop_xbound[1]],
                    [self.crop_ybound[0], self.crop_ybound[0]],
                    color='orange')
                self.mpl_canvas.axes.plot(
                    [self.crop_xbound[0], self.crop_xbound[1]],
                    [self.crop_ybound[1], self.crop_ybound[1]],
                    color='orange')
            
        # Tighten the border on the figure
        self.mpl_canvas.fig.tight_layout()
        self.mpl_canvas.draw()

        # UPDATE THE SIGNAL AXES
        # Grab the start and end indices
        start_i = self.axes_start_ind
        end_i = self.axes_end_ind+1
        for cnt, ind in enumerate(self.signal_coord):
            # Grab the canvas's attribute name
            canvasname = 'mpl_canvas_sig{}'.format(cnt+1)
            canvas = getattr(self, canvasname)
            # Clear axis for update
            canvas.axes.cla()
            canvas.draw()
            # Check to see if a signal has been selected
            if int(self.signal_toggle[cnt]) == 1:
                if self.ec_coupling_cb.isChecked() == True:
                    imaging_analysis = ImagingAnalysis.ImagingAnalysis()
                    all_data = imaging_analysis.ec_coupling_signal_load()

                    voltage_data = all_data[0]
                    calcium_data = all_data[1]
                    
                    # Plot the voltage signal
                    canvas.axes.plot(self.signal_time[start_i:end_i],
                                     voltage_data[start_i:end_i, ind[1], 
                                                  ind[0]], color='green')
                    # Plot the calcium signal on top of the voltage signal
                    canvas.axes.plot(self.signal_time[start_i:end_i],
                                     calcium_data[start_i:end_i, ind[1], 
                                                  ind[0]], color = 
                                     self.cnames[cnt])
                        
                    # Grab the min and max in the y-axis (just use one signal)
                    #edited this to make the max and min slightly larger and smaller
                    #because removing the first 5 tiffs makes them a bit too tall for the
                    #analysis window and we want to see the entire trace
                    vy0 = np.min(voltage_data[start_i:end_i, ind[1], 
                                              ind[0]])-0.05
                    vy1 = np.max(voltage_data[start_i:end_i, ind[1], 
                                              ind[0]])+0.05

                    cy0 = np.min(calcium_data[start_i:end_i, ind[1], 
                                              ind[0]])-0.05
                    cy1 = np.max(calcium_data[start_i:end_i, ind[1], 
                                              ind[0]])+0.05
                    
                    #searching for the highest max and the smallest min
                    if vy0 < cy0:
                        y0 = vy0
                        
                        if vy1 > cy1:
                            y1 = vy1
                        elif vy1 < cy1:
                            y1 = cy1
                        elif vy1 == cy1:
                            y1 = vy1
                            
                    elif vy0 > cy0: 
                        y0 = cy0
                        
                        if vy1 > cy1:
                            y1 = vy1
                        elif vy1 < cy1:
                            y1 = cy1
                        elif vy1 == cy1:
                            y1 = vy1
                            
                    elif vy0 == cy0:
                        y0 = vy0
                        
                        if vy1 > cy1:
                            y1 = vy1
                        elif vy1 < cy1:
                            y1 = cy1
                        elif vy1 == cy1:
                            y1 = vy1

                else:
                    # Plot the signal
                    canvas.axes.plot(self.signal_time[start_i:end_i],
                                     data[start_i:end_i, ind[1], ind[0]],
                                     color=self.cnames[cnt])
                    # Grab the min and max in the y-axis
                    #edited this to make the max and min slightly larger and smaller
                    #because removing the first 5 tiffs makes them a bit too tall for the
                    #analysis window and we want to see the entire trace
                    y0 = np.min(data[start_i:end_i, ind[1], ind[0]])-0.05
                    y1 = np.max(data[start_i:end_i, ind[1], ind[0]])+0.05
                    
                # Check for NAN values
                if np.isnan(y0) or np.isnan(y1):
                    y0 = -1.0
                    y1 = 1.0
                # Set y-axis limits
                canvas.axes.set_ylim(y0, y1)
                
                # Check to see if normalization has occurred
                if self.normalize_checkbox.isChecked():
                    # Get the position of the movie frame
                    x = self.signal_time[self.movie_scroll_obj.value()]
                    # Overlay the frame location of the play feature
                    canvas.axes.plot([x, x], [y0, y1], 'lime')
                    # Set the y-axis limits
                    canvas.axes.set_ylim(y0, y1)
                    
                # Check to see if limits have been established for analysis
                if self.analysis_bot_lim:
                    # Get the position of the lower limit marker
                    x = self.signal_time[self.anal_start_ind]
                    # Overlay the frame location of the play feature
                    canvas.axes.plot([x, x], [y0, y1], 'red')
                    # Set the y-axis limits
                    canvas.axes.set_ylim(y0, y1)
                    
                if self.analysis_top_lim:
                    # Get the position of the lower limit marker
                    x = self.signal_time[self.anal_end_ind]
                    # Overlay the frame location of the play feature
                    canvas.axes.plot([x, x], [y0, y1], 'red')
                    # Set the y-axis limits
                    canvas.axes.set_ylim(y0, y1)
                    # final = float(self.max_apd_edit.text())
                    
                if (self.analysis_drop.currentIndex() == 3 or 
                    self.analysis_drop.currentIndex() == 6):
                    
                    if self.analysis_bot_lim2:
                        # Get the position of the lower limit marker
                        x = self.signal_time[self.anal_start_ind2]
                        # Overlay the frame location of the play feature
                        canvas.axes.plot([x, x], [y0, y1], 'purple')
                        # Set the y-axis limits
                        canvas.axes.set_ylim(y0, y1)
                        
                    if self.analysis_top_lim2:
                        # Get the position of the lower limit marker
                        x = self.signal_time[self.anal_end_ind2]
                        # Overlay the frame location of the play feature
                        canvas.axes.plot([x, x], [y0, y1], 'purple')
                        # Set the y-axis limits
                        canvas.axes.set_ylim(y0, y1)
                        
                if self.analysis_y_lim:
                    # X-axis bounds
                    x0 = self.signal_time[start_i]
                    x1 = self.signal_time[end_i-1]
                    # Y-axis value
                    y = float(self.max_val_edit.text())
                    # Overlay the frame location of the play feature
                    canvas.axes.plot([x0, x1], [y, y], 'green')
                    # Set the x-axis limits
                    canvas.axes.set_xlim(self.signal_time[start_i],
                                         self.signal_time[end_i-1])
                # Set the x-axis limits
                canvas.axes.set_xlim(self.signal_time[start_i],
                                     self.signal_time[end_i-1])
                # Tighten the layout
                canvas.fig.tight_layout()
                # Draw the figure
                canvas.draw()

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=391, height=391, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        
class Stream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))

class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

if __name__ == '__main__':
    fig1 = Figure()
    ax1f1 = fig1.add_subplot(111)
    ax1f1.plot(np.random.rand(5))
    # create the GUI application
    app = QApplication(sys.argv)
    # instantiate and show the main window
    ks_main = MainWindow()
    # ks_main.addmpl(fig1)
    ks_main.show()
    # start the Qt main loop execution, exiting from this script
    # with the same return code as the Qt application
    sys.exit(app.exec_())
