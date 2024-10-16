import os
import sys
import time
import numpy as np
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QListWidget, QLabel, QFileDialog, QInputDialog,
                             QMessageBox, QProgressBar, QTabWidget, QTextEdit, QSplitter,
                             QComboBox, QSpinBox, QCheckBox, QGroupBox, QScrollArea, QDialog, QLineEdit, QListWidgetItem, QDialogButtonBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from pylsl import StreamInlet, resolve_stream
from EEGAnalysis import EEGAnalysis
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.offline import plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class EEGDataCollectionThread(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, action, data_dir):
        super().__init__()
        self.action = action
        self.data_dir = data_dir
        self.FFT_MAX_HZ = 60
        self.HM_SECONDS = 10
        self.TOTAL_ITERS = self.HM_SECONDS * 25

    def run(self):
        action_dir = os.path.join(self.data_dir, "raw_data", self.action)
        os.makedirs(action_dir, exist_ok=True)

        streams = resolve_stream('type', 'EEG')
        if not streams:
            self.update_log("No EEG stream available.")
            self.finished.emit()
            return
        inlet = StreamInlet(streams[0])
        channel_datas = []

        for i in range(self.TOTAL_ITERS):
            channel_data = []
            for _ in range(16):
                sample, timestamp = inlet.pull_sample()
                channel_data.append(sample[:self.FFT_MAX_HZ])
            channel_datas.append(channel_data)
            self.update_progress.emit(int((i + 1) / self.TOTAL_ITERS * 100))

        filename = os.path.join(action_dir, f"{int(time.time())}.npy")
        np.save(filename, np.array(channel_datas))
        self.finished.emit()

class DataProcessingThread(QThread):
    finished = pyqtSignal()

    def __init__(self, eeg_analysis):
        super().__init__()
        self.eeg_analysis = eeg_analysis

    def run(self):
        self.eeg_analysis.check_data_consistency()
        self.eeg_analysis.clean_and_balance_data()
        self.eeg_analysis.check_data_consistency()
        self.finished.emit()

class PreprocessingThread(QThread):
    progress_update = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, framework, data_dir):
        super().__init__()
        self.framework = framework
        self.data_dir = data_dir

    def run(self):
        if 'cleaned_data' not in self.framework.data or not self.framework.data['cleaned_data']:
            self.error.emit("No cleaned data available. Please clean the data before preprocessing.")
            self.finished.emit()
            return

        total_actions = len(self.framework.actions)
        processed_actions = 0
        for action in self.framework.actions:
            if action in self.framework.data['cleaned_data']:
                try:
                    data = self.framework.data['cleaned_data'][action]
                    self.framework.preprocess_eeg(data, action)
                    processed_actions += 1
                    progress = int((processed_actions / total_actions) * 100)
                    self.progress_update.emit(progress)
                except Exception as e:
                    self.error.emit(f"Error preprocessing action '{action}': {str(e)}")
            else:
                self.error.emit(f"No cleaned data found for action '{action}'. Skipping.")
        
        if processed_actions == 0:
            self.error.emit("No actions could be preprocessed. Please check your cleaned data.")
        else:
            self.framework.data['preprocessed_data'] = self.framework.preprocessed_data
        self.finished.emit()
        
class EEGAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Analysis Framework")
        self.setGeometry(100, 100, 1600, 900)
        self.base_dir = os.getcwd()
        self.data_dir = os.path.join(self.base_dir, "model_data")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.framework = EEGAnalysis(self.base_dir)
        self.framework.progress_update.connect(self.update_log)

        self.data_lists = {}
        self.selected_channels = list(range(16))  # All channels selected by default
        self.selected_frequencies = list(range(1, 61))  # All frequencies selected by default
        self.init_ui()
        self.update_data_lists()
        self.load_recent_logs()
        self.load_and_visualize_data()
        self.update_visualization()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side: controls and data lists
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Data selection controls
        control_group = QGroupBox("Data Selection")
        control_layout = QVBoxLayout()

        self.action_combo = QComboBox()
        self.action_combo.currentTextChanged.connect(self.update_visualization)
        control_layout.addWidget(QLabel("Action:"))
        control_layout.addWidget(self.action_combo)

        self.sample_spinner = QSpinBox()
        self.sample_spinner.valueChanged.connect(self.update_visualization)
        control_layout.addWidget(QLabel("Sample:"))
        control_layout.addWidget(self.sample_spinner)

        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)

        # Channel selection
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QVBoxLayout()
        self.channel_checkboxes = []
        for i in range(16):
            cb = QCheckBox(f"Channel {i+1}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_channel_selection)
            channel_layout.addWidget(cb)
            self.channel_checkboxes.append(cb)
        channel_group.setLayout(channel_layout)
        channel_scroll = QScrollArea()
        channel_scroll.setWidget(channel_group)
        channel_scroll.setWidgetResizable(True)
        left_layout.addWidget(channel_scroll)
        
        # Add frequency selection button
        self.freq_select_btn = QPushButton("Select Frequencies")
        self.freq_select_btn.clicked.connect(self.open_frequency_selection)
        control_layout.addWidget(self.freq_select_btn)

        # Tabs for different data stages
        self.tabs = QTabWidget()
        self.tabs.clear()
        for data_type in ["raw_data", "cleaned_data", "preprocessed_data", "validation_data"]:
            self.tabs.addTab(self.create_data_tab(data_type), data_type.replace('_', ' ').title())
        self.tabs.currentChanged.connect(self.on_tab_changed)
        left_layout.addWidget(self.tabs)

        # Data collection and processing buttons
        button_layout = QHBoxLayout()
        self.collect_data_btn = QPushButton("Collect EEG Data")
        self.collect_data_btn.clicked.connect(self.collect_eeg_data)
        button_layout.addWidget(self.collect_data_btn)

        self.clean_data_btn = QPushButton("Clean and Balance Data")
        self.clean_data_btn.clicked.connect(self.clean_and_balance_data)
        button_layout.addWidget(self.clean_data_btn)

        self.preprocess_data_btn = QPushButton("Preprocess Data")
        self.preprocess_data_btn.clicked.connect(self.preprocess_data)
        button_layout.addWidget(self.preprocess_data_btn)

        self.analyze_btn = QPushButton("Run Analysis")
        self.analyze_btn.clicked.connect(self.run_analysis)
        button_layout.addWidget(self.analyze_btn)

        left_layout.addLayout(button_layout)

        # Add Compare Data button
        self.compare_data_btn = QPushButton("Compare Data")
        self.compare_data_btn.clicked.connect(self.compare_data)
        left_layout.addWidget(self.compare_data_btn)

        # Add combo boxes for data comparison
        self.comparison_layout = QHBoxLayout()
        self.comparison_layout.addWidget(QLabel("Compare:"))
        self.compare_from_combo = QComboBox()
        self.compare_from_combo.addItems(["Raw Data", "Cleaned Data", "Preprocessed Data"])
        self.comparison_layout.addWidget(self.compare_from_combo)
        self.comparison_layout.addWidget(QLabel("with:"))
        self.compare_to_combo = QComboBox()
        self.compare_to_combo.addItems(["Raw Data", "Cleaned Data", "Preprocessed Data"])
        self.comparison_layout.addWidget(self.compare_to_combo)
        left_layout.addLayout(self.comparison_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)

        # Right side: visualization and logs
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 2D Visualization
        self.plot_2d = EEGPlot(self)
        right_layout.addWidget(self.plot_2d, 3)

        # 2D Explanation
        self.explanation_2d = QTextEdit()
        self.explanation_2d.setReadOnly(True)
        right_layout.addWidget(self.explanation_2d, 1)

        # 3D Visualization and Explanation
        vis_3d_layout = QHBoxLayout()

        self.fig_3d = Figure(figsize=(8, 6))
        self.canvas_3d = FigureCanvas(self.fig_3d)
        vis_3d_layout.addWidget(self.canvas_3d, 2)

        self.explanation_3d = QTextEdit()
        self.explanation_3d.setReadOnly(True)
        vis_3d_layout.addWidget(self.explanation_3d, 1)

        right_layout.addLayout(vis_3d_layout, 2)

        # Log section
        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        log_button_layout = QHBoxLayout()
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        log_button_layout.addWidget(self.clear_log_btn)
        self.save_log_btn = QPushButton("Save Log")
        self.save_log_btn.clicked.connect(self.save_log)
        log_button_layout.addWidget(self.save_log_btn)
        log_layout.addLayout(log_button_layout)

        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group, 1)

        # Add splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 1200])

        main_layout.addWidget(splitter)

    def update_channel_selection(self):
        self.selected_channels = [i for i in range(16) if self.channel_checkboxes[i].isChecked()]
        self.update_visualization()

    def create_data_tab(self, data_type):
        tab = QWidget()
        tab.setObjectName(data_type)
        layout = QVBoxLayout(tab)
        self.data_lists[data_type] = QListWidget()
        layout.addWidget(self.data_lists[data_type])
        return tab

    def update_data_lists(self):
        for data_type in self.framework.data.keys():
            list_widget = self.data_lists[data_type]
            list_widget.clear()
            for action in self.framework.data[data_type].keys():
                list_widget.addItem(action)

    def on_tab_changed(self, index):
        self.cleanup_animation()
        tab_name = self.tabs.tabText(index)
        self.visualize_data_for_tab(tab_name)
        self.update_visualization()  

    def cleanup_animation(self):
        if hasattr(self, 'anim') and self.anim is not None:
            self.anim.event_source.stop()
            self.anim._stop()
            del self.anim
        if hasattr(self, 'ax_3d'):
            self.ax_3d = None
        if hasattr(self, 'fig_3d'):
            self.fig_3d.clf()
        if hasattr(self, 'canvas_3d'):
            self.canvas_3d.draw()
    
    def load_and_visualize_data(self):
        print("Loading and visualizing data...")
        self.framework.load_data()

        if not self.framework.data:
            self.update_log("No data available for visualization.")
            return

        # Populate action combo box
        self.action_combo.clear()
        self.action_combo.addItems(sorted(self.framework.actions))

        # Set sample spinner range
        max_samples = 0
        for data_type in self.framework.data.keys():
            for action in self.framework.data[data_type].keys():
                max_samples = max(max_samples, len(self.framework.data[data_type][action]))

        if max_samples > 0:
            self.sample_spinner.setRange(0, max_samples - 1)
        else:
            self.sample_spinner.setRange(0, 0)
            self.update_log("No samples found in the dataset.")
            return

        # Determine which tab to display based on available data
        if self.framework.data['preprocessed_data']:
            self.tabs.setCurrentIndex(2)  # Switch to "Preprocessed Data" tab
            self.visualize_data_for_tab("Preprocessed Data")
        elif self.framework.data['cleaned_data']:
            self.tabs.setCurrentIndex(1)  # Switch to "Cleaned Data" tab
            self.visualize_data_for_tab("Cleaned Data")
        elif self.framework.data['raw_data']:
            self.tabs.setCurrentIndex(0)  # Switch to "Raw Data" tab
            self.visualize_data_for_tab("Raw Data")
        else:
            self.update_log("No data available for visualization in any category.")

        self.update_data_lists()
        self.update_visualization()

    def visualize_data_for_tab(self, data_type):
        # Convert data_type to the format used in self.data dictionary
        data_type_key = data_type.lower().replace(' ', '_')
        
        data = self.framework.get_data_for_visualization(data_type_key)
        if data is None:
            self.clear_visualizations()
            return

        action = self.action_combo.currentText()
        if not action or action not in data:
            self.clear_visualizations()
            return

        sample_index = min(self.sample_spinner.value(), len(data[action]) - 1)
        self.plot_2d.plot_eeg(data, action, sample_index, data_type, 
                              self.selected_channels, self.selected_frequencies)
        self.update_3d_animation(data, action, sample_index, data_type)
        self.update_explanations(data[action][sample_index], action)
        
    def update_explanations(self, sample, action):
        # Update 2D explanation
        explanation_2d = self.generate_2d_explanation(sample, action)
        self.explanation_2d.setText(explanation_2d)

        # Update 3D explanation
        explanation_3d = self.generate_3d_explanation(sample, action)
        self.explanation_3d.setText(explanation_3d)
        
    def open_frequency_selection(self):
        dialog = FrequencySelectionDialog(self.selected_frequencies)
        if dialog.exec_():
            self.selected_frequencies = dialog.get_selected_frequencies()
            self.update_visualization()
    
    def update_3d_animation(self, data, action, sample_index, data_type):
        if action not in data or sample_index >= len(data[action]):
            self.update_log(f"Invalid action or sample index for {data_type} data.")
            return

        sample = data[action][sample_index]

        self.cleanup_animation()  # Ensure any existing animation is cleaned up
        self.fig_3d.clear()
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')

        channel_coords = [
            (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
            (-0.5, 0, 0.75), (0.5, 0, 0.75),
            (-0.75, -0.5, 0.5), (0.75, -0.5, 0.5),
            (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
            (-0.5, -1, 0.5), (0.5, -1, 0.5),
            (-0.75, -1.5, 0), (0.75, -1.5, 0),
            (0, 0.75, 0.5), (0, -0.75, 0.75),
            (-0.25, -0.25, 1), (0.25, -0.25, 1)
        ]

        x, y, z = zip(*channel_coords)
        
        # Calculate the mean amplitude across selected frequencies
        mean_amplitude = np.mean(sample[:, :, [f-1 for f in self.selected_frequencies]], axis=2)
        
        self.scatter = self.ax_3d.scatter(x, y, z, c=mean_amplitude[0], cmap='viridis', s=100)
        self.fig_3d.colorbar(self.scatter)

        center = (0, 0, 0)
        for i, (x, y, z) in enumerate(channel_coords):
            if i in self.selected_channels:
                self.ax_3d.plot([center[0], x], [center[1], y], [center[2], z], color='gray', alpha=0.5)

        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title(f'3D EEG Animation - {action.capitalize()} ({data_type})')

        def update(frame):
            if self.ax_3d is None:
                return (self.scatter,)
            colors = mean_amplitude[frame]
            sizes = 100 + 200 * (colors - colors.min()) / (colors.max() - colors.min())
            self.scatter._sizes = sizes
            self.scatter.set_array(colors)
            return (self.scatter,)

        self.anim = FuncAnimation(self.fig_3d, update, frames=range(sample.shape[0]),
                                  interval=50, blit=True)
        self.canvas_3d.draw()

        explanation = self.generate_3d_explanation(sample, action)
        self.explanation_3d.setText(explanation)

    def update_action_combo(self):
        self.action_combo.clear()
        self.action_combo.addItems(sorted(self.framework.actions))

    def update_visualization(self):
        current_tab = self.tabs.currentWidget()
        if current_tab:
            data_type = self.tabs.tabText(self.tabs.currentIndex())
            self.visualize_data_for_tab(data_type)

    def clear_visualizations(self):
        self.cleanup_animation()
        self.plot_2d.canvas.figure.clear()
        self.plot_2d.canvas.draw()
        self.explanation_2d.clear()
        self.explanation_3d.clear()

    def collect_eeg_data(self):
        action, ok = QInputDialog.getText(self, "Collect EEG Data", "Enter action name:")
        if ok and action:
            self.data_collection_thread = EEGDataCollectionThread(action, self.data_dir)
            self.data_collection_thread.update_progress.connect(self.progress_bar.setValue)
            self.data_collection_thread.finished.connect(self.data_collection_finished)
            self.data_collection_thread.start()
            self.collect_data_btn.setEnabled(False)

    def data_collection_finished(self):
        self.collect_data_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.update_data_lists()
        QMessageBox.information(self, "Success", "EEG data collection completed.")

    def clean_and_balance_data(self):
        if not self.framework.data['raw_data']:
            QMessageBox.warning(self, "No Data", "No raw data available. Please load data first.")
            return

        self.clean_data_btn.setEnabled(False)
        self.processing_thread = DataProcessingThread(self.framework)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def on_processing_finished(self):
        self.clean_data_btn.setEnabled(True)
        self.update_data_lists()
        QMessageBox.information(self, "Success", "Data cleaned, balanced, and consistency checked.")
        
    def save_preprocessing_settings(self):
        settings = {action: self.framework.preprocessing_settings.get(action, {}) 
                    for action in self.framework.actions}
        with open('preprocessing_settings.json', 'w') as f:
            json.dump(settings, f)

    def load_preprocessing_settings(self):
        try:
            with open('preprocessing_settings.json', 'r') as f:
                settings = json.load(f)
            for action, action_settings in settings.items():
                self.framework.set_preprocessing_settings(action, action_settings)
        except FileNotFoundError:
            pass  # No settings file found, will use defaults


    def preprocess_data(self):
        if not self.framework.data.get('cleaned_data'):
            QMessageBox.warning(self, "No Cleaned Data", "No cleaned data available. Please clean the data first.")
            return

        self.preprocess_data_btn.setEnabled(False)
        
        dialog = MultiActionPreprocessingSettingsDialog(self.framework.actions, self.framework.preprocessing_settings, self)
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            new_settings = dialog.get_settings()
            for action, settings in new_settings.items():
                self.framework.set_preprocessing_settings(action, settings)
            self.save_preprocessing_settings()
            
            # Proceed with preprocessing
            self.preprocessing_thread = PreprocessingThread(self.framework, self.data_dir)
            self.preprocessing_thread.progress_update.connect(self.progress_bar.setValue)
            self.preprocessing_thread.finished.connect(self.preprocessing_finished)
            self.preprocessing_thread.error.connect(self.preprocessing_error)
            self.preprocessing_thread.start()
        else:
            self.preprocess_data_btn.setEnabled(True)
    def preprocessing_finished(self):
        self.preprocess_data_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.update_data_lists()
        self.tabs.setCurrentIndex(self.tabs.indexOf(self.tabs.findChild(QWidget, "preprocessed_data")))
        QMessageBox.information(self, "Success", "Data preprocessing completed.")

    def run_analysis(self):
        self.framework.run_analysis()
        results = self.framework.get_results()
        QMessageBox.information(self, "Analysis Complete", "EEG analysis completed. Check the logs for detailed results.")

    def update_log(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def clear_log(self):
        self.log_text.clear()

    def save_log(self):
        log_content = self.log_text.toPlainText()
        if log_content:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Log File", "", "Log Files (*.log);;All Files (*)")
            if file_name:
                with open(file_name, 'w') as f:
                    f.write(log_content)
                QMessageBox.information(self, "Success", "Log file saved successfully.")
        else:
            QMessageBox.warning(self, "Empty Log", "The log is empty. Nothing to save.")

    def load_recent_logs(self):
        log_dir = os.path.join(os.path.dirname(self.data_dir), 'logs')
        if os.path.exists(log_dir):
            log_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.log')], reverse=True)
            if log_files:
                most_recent_log = os.path.join(log_dir, log_files[0])
                with open(most_recent_log, 'r') as f:
                    self.log_text.setText(f.read())

    def ensure_data_directories(self):
        for subdir in ["raw_data", "cleaned_data", "preprocessed_data", "validation_data", "models"]:
            os.makedirs(os.path.join(self.data_dir, subdir), exist_ok=True)

    def generate_2d_explanation(self, sample, action):
        explanation = f"2D EEG Visualization for {action.capitalize()}:\n\n"
        explanation += "This plot shows the EEG signal for each of the selected channels over time.\n"
        explanation += "Each subplot represents a different channel, with time on the x-axis and amplitude on the y-axis.\n"
        explanation += f"Displaying frequencies: {', '.join(map(str, self.selected_frequencies))} Hz\n\n"

        if not self.selected_channels:
            explanation += "No channels are currently selected. Please select at least one channel to view data.\n"
            return explanation

        mean_amplitudes = np.mean(sample[:, self.selected_channels, :][:, :, [f-1 for f in self.selected_frequencies]], axis=(0, 2))
        
        if len(mean_amplitudes) > 0:
            max_channel_index = np.argmax(mean_amplitudes)
            min_channel_index = np.argmin(mean_amplitudes)
            max_channel = self.selected_channels[max_channel_index]
            min_channel = self.selected_channels[min_channel_index]

            explanation += f"Channel {max_channel + 1} shows the highest average activity.\n"
            explanation += f"Channel {min_channel + 1} shows the lowest average activity.\n\n"
        else:
            explanation += "No data available for the selected channels and frequencies.\n\n"

        for band, (low, high) in self.framework.frequency_bands.items():
            band_freqs = [f for f in self.selected_frequencies if low <= f <= high]
            if band_freqs:
                band_power = np.mean(sample[:, self.selected_channels, :][:, :, [f-1 for f in band_freqs]])
                explanation += f"{band.capitalize()} band power: {band_power:.2f}\n"

        return explanation

    def generate_3d_explanation(self, sample, action):
        explanation = f"3D EEG Visualization for {action.capitalize()}:\n\n"
        explanation += "This animation shows how EEG activity changes over time across the scalp.\n"
        explanation += "Each point represents an electrode, with color and size indicating the amplitude of the signal.\n"
        explanation += f"Displaying average activity for frequencies: {', '.join(map(str, self.selected_frequencies))} Hz\n\n"

        mean_amplitudes = np.mean(sample[:, :, [f-1 for f in self.selected_frequencies]], axis=(0, 2))
        max_channel = np.argmax(mean_amplitudes)
        min_channel = np.argmin(mean_amplitudes)

        explanation += f"Channel {max_channel + 1} shows the highest average activity over time.\n"
        explanation += f"Channel {min_channel + 1} shows the lowest average activity over time.\n\n"

        explanation += "Observed patterns:\n"
        explanation += "- Look for color and size changes that propagate across the scalp.\n"
        explanation += "- Notice any rhythmic fluctuations in specific regions.\n"
        explanation += "- Pay attention to sudden, widespread changes in activity.\n"
        explanation += "- The lines connecting to the center represent the signal path for each channel.\n"

        return explanation

    def compare_data(self):
        from_data = self.compare_from_combo.currentText().lower().replace(" ", "_")
        to_data = self.compare_to_combo.currentText().lower().replace(" ", "_")
        
        if from_data == to_data:
            QMessageBox.warning(self, "Invalid Selection", "Please select different data types for comparison.")
            return

        self.plot_data_comparison(from_data, to_data)

    def plot_data_comparison(self, from_data, to_data):
        action = self.action_combo.currentText()
        sample_index = self.sample_spinner.value()

        # Check if the data types exist
        if from_data not in self.framework.data or to_data not in self.framework.data:
            QMessageBox.warning(self, "Data Not Available", f"One or both of the selected data types ({from_data}, {to_data}) are not available.")
            return

        # Check if the action exists in both data types
        if action not in self.framework.data[from_data] or action not in self.framework.data[to_data]:
            QMessageBox.warning(self, "Action Not Available", f"The selected action '{action}' is not available in one or both of the selected data types.")
            return

        # Check if the sample index is valid
        if sample_index >= len(self.framework.data[from_data][action]) or sample_index >= len(self.framework.data[to_data][action]):
            QMessageBox.warning(self, "Invalid Sample Index", f"The selected sample index {sample_index} is out of range for one or both of the selected data types.")
            return

        # Access the data
        from_sample = self.framework.data[from_data][action][sample_index]
        to_sample = self.framework.data[to_data][action][sample_index]

        fig, axs = plt.subplots(4, 4, figsize=(15, 15))
        fig.suptitle(f'Comparison: {from_data.capitalize()} vs {to_data.capitalize()} - {action} - Sample {sample_index}')

        for i, ax in enumerate(axs.flat):
            if i < 16:  # We have 16 channels
                ax.plot(from_sample[:, i], label=from_data.capitalize(), alpha=0.7)
                ax.plot(to_sample[:, i], label=to_data.capitalize(), alpha=0.7)
                ax.set_title(f'Channel {i+1}')
                ax.set_ylim(min(np.min(from_sample), np.min(to_sample)), max(np.max(from_sample), np.max(to_sample)))
                if i == 0:  # Only show legend for the first subplot
                    ax.legend()

        plt.tight_layout()
        plt.show()

    def closeEvent(self, event):
        if hasattr(self, 'anim'):
            self.anim.event_source.stop()
        super().closeEvent(event)
        
class FrequencySelectionDialog(QDialog):
    def __init__(self, current_frequencies):
        super().__init__()
        self.setWindowTitle("Select Frequencies")
        self.setGeometry(300, 300, 400, 300)

        layout = QVBoxLayout()

        self.freq_input = QLineEdit()
        self.freq_input.setPlaceholderText("Enter frequency ranges (e.g., 1-5, 10-20)")
        layout.addWidget(self.freq_input)

        self.freq_list = QListWidget()
        for freq in range(1, 61):
            item = QListWidgetItem(f"{freq} Hz")
            item.setCheckState(Qt.Checked if freq in current_frequencies else Qt.Unchecked)
            self.freq_list.addItem(item)
        layout.addWidget(self.freq_list)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

        self.freq_input.textChanged.connect(self.update_selection_from_input)

    def update_selection_from_input(self, text):
        ranges = text.split(',')
        selected_freqs = set()
        for r in ranges:
            try:
                if '-' in r:
                    start, end = map(int, r.split('-'))
                    selected_freqs.update(range(start, end + 1))
                else:
                    selected_freqs.add(int(r))
            except ValueError:
                continue

        for i in range(self.freq_list.count()):
            item = self.freq_list.item(i)
            freq = i + 1
            item.setCheckState(Qt.Checked if freq in selected_freqs else Qt.Unchecked)

    def get_selected_frequencies(self):
        return [i + 1 for i in range(self.freq_list.count()) if self.freq_list.item(i).checkState() == Qt.Checked]
    
class EEGPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.scroll = QScrollArea(self)
        self.layout.addWidget(self.scroll)
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.canvas = FigureCanvas(Figure(figsize=(12, 8)))
        self.plot_layout.addWidget(self.canvas)
        self.scroll.setWidget(self.plot_widget)
        self.scroll.setWidgetResizable(False)
        self.axes = []

    def plot_eeg(self, data, action, sample_index, data_type, selected_channels, selected_frequencies):
        self.canvas.figure.clear()
        self.axes = []

        num_channels = len(selected_channels)
        num_cols = 4
        num_rows = (num_channels + num_cols - 1) // num_cols

        # Adjust figure size based on the number of rows
        fig_height = 3 * num_rows + 1  # 3 inches per row, plus 1 inch for title and spacing
        self.canvas.figure.set_size_inches(12, fig_height)

        # Create subplot grid with adjusted spacing
        gs = self.canvas.figure.add_gridspec(nrows=num_rows, ncols=num_cols, 
                                             hspace=0.5,    # Increase space between rows
                                             wspace=0.3,    # Add some space between columns
                                             top=0.92,      # Adjust top margin for title
                                             bottom=0.05,   # Adjust bottom margin
                                             left=0.1,      # Adjust left margin
                                             right=0.95)    # Adjust right margin

        for i, channel in enumerate(selected_channels):
            row = i // num_cols
            col = i % num_cols
            ax = self.canvas.figure.add_subplot(gs[row, col])
            channel_data = data[action][sample_index][:, channel, :]
            for freq in selected_frequencies:
                ax.plot(channel_data[:, freq-1], label=f'{freq} Hz')
            ax.set_title(f'Channel {channel + 1}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.tick_params(axis='both', which='major', labelsize=8)
            self.axes.append(ax)

        self.canvas.figure.suptitle(f'{data_type} EEG Data - {action.capitalize()} - Sample {sample_index}',
                                    fontsize=16, y=0.98)  # Adjust title position
        self.canvas.draw()

class MultiActionPreprocessingSettingsDialog(QDialog):
    def __init__(self, actions, current_settings, parent=None):
        super().__init__(parent)
        self.actions = actions
        self.current_settings = current_settings
        self.new_settings = {}
        self.setWindowTitle("Preprocessing Settings")
        self.init_ui()

    def get_default_preprocessing_settings(self):
        return {
            'lowcut': 1,
            'highcut': 60,
            'notch_freq': 50.0,
            'quality_factor': 30.0,
            'adaptive_mu': 0.01,
            'adaptive_order': 5,
            'ica_components': 16
        }

    def init_ui(self):
        layout = QVBoxLayout()
        
        self.tab_widget = QTabWidget()
        for action in self.actions:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            settings = self.current_settings.get(action, self.get_default_preprocessing_settings())
            
            for key, value in settings.items():
                hlayout = QHBoxLayout()
                hlayout.addWidget(QLabel(f"{key}:"))
                line_edit = QLineEdit(str(value))
                line_edit.setObjectName(f"{action}_{key}")
                hlayout.addWidget(line_edit)
                tab_layout.addLayout(hlayout)
            
            self.tab_widget.addTab(tab, action)
        
        layout.addWidget(self.tab_widget)

        button_box = QDialogButtonBox()
        self.save_button = button_box.addButton("Save", QDialogButtonBox.ActionRole)
        self.continue_button = button_box.addButton("Save and Continue", QDialogButtonBox.AcceptRole)
        self.cancel_button = button_box.addButton(QDialogButtonBox.Cancel)

        self.save_button.clicked.connect(self.save_settings)
        self.continue_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        layout.addWidget(button_box)
        self.setLayout(layout)

    def save_settings(self):
        self.update_settings()
        QMessageBox.information(self, "Settings Saved", "Preprocessing settings have been saved.")

    def update_settings(self):
        self.new_settings = {}
        for i, action in enumerate(self.actions):
            tab = self.tab_widget.widget(i)
            settings = {}
            for key in self.get_default_preprocessing_settings().keys():
                line_edit = tab.findChild(QLineEdit, f"{action}_{key}")
                if line_edit:
                    settings[key] = float(line_edit.text())
            self.new_settings[action] = settings

    def accept(self):
        self.update_settings()
        super().accept()

    def get_settings(self):
        return self.new_settings

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = EEGAnalysisGUI()
    gui.show()
    sys.exit(app.exec_())
