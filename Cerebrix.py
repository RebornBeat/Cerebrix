import os
import sys
import time
import numpy as np
import json
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRunnable, QObject, QThreadPool, QStringListModel, QEvent
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QListWidget, QLabel, QFileDialog, QInputDialog,
                             QMessageBox, QProgressBar, QTabWidget, QTextEdit, QSplitter,
                             QComboBox, QSpinBox, QCheckBox, QGroupBox, QScrollArea, QDialog,
                             QLineEdit, QListWidgetItem, QDialogButtonBox, QListView, QCompleter)
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from pylsl import StreamInlet, resolve_stream
from EEGAnalysis import EEGAnalysis
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from data_manager import DataManager
from concurrent.futures import ThreadPoolExecutor
    
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
        
class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                self.signals.progress.emit(result[0])
                result = result[1]
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()

class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(object)
    progress = pyqtSignal(str)
    
from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QListView
from PyQt5.QtGui import QStandardItemModel, QStandardItem

class SearchWidget(QWidget):
    def __init__(self, main_gui, parent=None):
        super().__init__(parent)
        self.main_gui = main_gui
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.search_field = QLineEdit(self)
        self.search_field.setPlaceholderText("Search for sample file")
        self.layout.addWidget(self.search_field)

        self.dropdown = QListView(self)
        self.dropdown.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.dropdown.setEditTriggers(QListView.NoEditTriggers)
        self.dropdown.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.dropdown.setMouseTracking(True)
        self.dropdown.clicked.connect(self.on_item_clicked)
        self.dropdown.entered.connect(self.on_item_entered)

        self.model = QStandardItemModel()
        self.dropdown.setModel(self.model)

        self.search_field.textChanged.connect(self.on_text_changed)
        self.search_field.installEventFilter(self)
        self.dropdown.installEventFilter(self)

        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_dropdown)

    def on_text_changed(self):
        # Reset and start the timer
        self.update_timer.stop()
        self.update_timer.start(600)  # 200ms delay

    def update_dropdown(self):
        text = self.search_field.text()
        self.model.clear()
        if text:
            for item in self.main_gui.get_filtered_samples(text):
                self.model.appendRow(QStandardItem(item))
            
            if self.model.rowCount() > 0:
                self.show_dropdown()
            else:
                self.hide_dropdown()
        else:
            self.hide_dropdown()

    def show_dropdown(self):
        width = self.search_field.width()
        height = min(200, self.model.rowCount() * 20)
        x = self.search_field.mapToGlobal(self.search_field.rect().bottomLeft()).x()
        y = self.search_field.mapToGlobal(self.search_field.rect().bottomLeft()).y()
        
        self.dropdown.setGeometry(x, y, width, height)
        self.dropdown.show()

        # Highlight the first item
        if self.model.rowCount() > 0:
            first_item = self.model.item(0)
            self.dropdown.setCurrentIndex(first_item.index())

    def hide_dropdown(self):
        self.dropdown.hide()

    def on_item_clicked(self, index):
        item = self.model.itemFromIndex(index)
        if item:
            self.search_field.setText(item.text())
            self.hide_dropdown()
            self.main_gui.select_sample_from_search(item.text())

    def on_item_entered(self, index):
        self.dropdown.setCurrentIndex(index)

    def eventFilter(self, obj, event):
        if obj == self.search_field:
            if event.type() == QEvent.KeyPress:
                if self.dropdown.isVisible():
                    current_index = self.dropdown.currentIndex()
                    if event.key() == Qt.Key_Down:
                        next_index = self.model.index(current_index.row() + 1, 0)
                        if next_index.isValid():
                            self.dropdown.setCurrentIndex(next_index)
                        return True
                    elif event.key() == Qt.Key_Up:
                        prev_index = self.model.index(current_index.row() - 1, 0)
                        if prev_index.isValid():
                            self.dropdown.setCurrentIndex(prev_index)
                        return True
                    elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                        self.on_item_clicked(current_index)
                        return True
                    elif event.key() == Qt.Key_Escape:
                        self.hide_dropdown()
                        return True
        elif obj == self.dropdown:
            if event.type() == QEvent.MouseButtonPress:
                if not self.dropdown.rect().contains(event.pos()):
                    self.hide_dropdown()
                    return True
        return super().eventFilter(obj, event)

    def focusOutEvent(self, event):
        # Use a timer to delay hiding the dropdown
        QTimer.singleShot(100, self.check_focus_and_hide_dropdown)
        super().focusOutEvent(event)

    def check_focus_and_hide_dropdown(self):
        # Only hide the dropdown if neither the search field nor the dropdown has focus
        if not self.search_field.hasFocus() and not self.dropdown.hasFocus():
            self.hide_dropdown()

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
        self.data_lists = {}  # Initialize data_lists here
        self.selected_channels = list(range(16))  # All channels selected by default
        self.selected_frequencies = list(range(1, 61))  # All frequencies selected by default
        self.threadpool = QThreadPool()
        self.init_ui()
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress_from_queue)
        self.progress_timer.start(10)  

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

        # Action selection
        action_layout = QHBoxLayout()
        action_layout.addWidget(QLabel("Action:"))
        self.action_combo = QComboBox()
        self.action_combo.currentTextChanged.connect(self.update_visualization)
        action_layout.addWidget(self.action_combo)
        control_layout.addLayout(action_layout)

        # File name display
        self.file_name_label = QLabel()
        control_layout.addWidget(self.file_name_label)

        # Sample selection
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("Sample:"))
        self.sample_spinner = QSpinBox()
        self.sample_spinner.valueChanged.connect(self.update_visualization)
        sample_layout.addWidget(self.sample_spinner)
        control_layout.addLayout(sample_layout)

        # New SearchWidget
        self.search_widget = SearchWidget(self, parent=control_group)
        control_layout.addWidget(self.search_widget)
        self.search_widget.search_field.installEventFilter(self.search_widget)

        # Frequency selection
        self.freq_select_btn = QPushButton("Select Frequencies")
        self.freq_select_btn.clicked.connect(self.open_frequency_selection)
        control_layout.addWidget(self.freq_select_btn)

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

        # Tabs for different data stages
        self.tabs = QTabWidget()
        self.tabs.clear()
        for data_type in ["raw_data", "cleaned_data", "validation_data"]:
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

        self.train_data_btn = QPushButton("Train Data")
        self.train_data_btn.clicked.connect(self.train_data)
        button_layout.addWidget(self.train_data_btn)

        left_layout.addLayout(button_layout)

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
        self.load_recent_logs()
        self.load_data_async()

    def update_progress_from_queue(self):
        while not self.framework.progress_queue.empty():
            progress, message = self.framework.progress_queue.get()
            self.progress_bar.setValue(progress)
            if message:
                self.update_log(message)
        QApplication.processEvents()  # Force GUI update

        
    def load_data_async(self):
        self.update_log("Loading data...")
        worker = Worker(self.framework.load_data, progress_callback=self.update_progress)
        worker.signals.result.connect(self.on_data_loaded)
        worker.signals.error.connect(self.on_worker_error)
        self.threadpool.start(worker)
        
    def on_data_loaded(self):
        self.update_action_combo()
        self.update_sample_spinner()
        self.update_data_lists()
        self.load_and_visualize_data()

    def update_channel_selection(self):
        self.selected_channels = [i for i in range(16) if self.channel_checkboxes[i].isChecked()]
        self.update_visualization()
        
    def get_filtered_samples(self, text):
        current_tab = self.tabs.currentWidget()
        if current_tab:
            data_type = self.tabs.tabText(self.tabs.currentIndex()).lower().replace(' ', '_')
            action = self.action_combo.currentText()
            samples = self.framework.data_manager.data_catalog[data_type][action]
            return [os.path.basename(s) for s in samples if text.lower() in os.path.basename(s).lower()]
        return []

    def select_sample_from_search(self, sample_name):
        current_tab = self.tabs.currentWidget()
        if current_tab:
            data_type = self.tabs.tabText(self.tabs.currentIndex()).lower().replace(' ', '_')
            action = self.action_combo.currentText()
            samples = self.framework.data_manager.data_catalog[data_type][action]
            for i, sample_path in enumerate(samples):
                if os.path.basename(sample_path) == sample_name:
                    self.sample_spinner.setValue(i)
                    break
        self.update_visualization()

    def update_file_name_display(self):
        current_tab = self.tabs.currentWidget()
        if current_tab:
            data_type = self.tabs.tabText(self.tabs.currentIndex()).lower().replace(' ', '_')
            action = self.action_combo.currentText()
            sample_index = self.sample_spinner.value()
            if action in self.framework.data_manager.data_catalog[data_type]:
                if sample_index < len(self.framework.data_manager.data_catalog[data_type][action]):
                    file_path = self.framework.data_manager.data_catalog[data_type][action][sample_index]
                    file_name = os.path.basename(file_path)
                    self.file_name_label.setText(f"File: {file_name}")
                    return
        self.file_name_label.setText("File: N/A")

    def create_data_tab(self, data_type):
        tab = QWidget()
        tab.setObjectName(data_type)
        layout = QVBoxLayout(tab)
        self.data_lists[data_type] = QListWidget()
        layout.addWidget(self.data_lists[data_type])
        return tab

    def update_data_lists(self):
        for data_type in ['raw_data', 'cleaned_data', 'validation_data']:
            list_widget = self.data_lists[data_type]
            list_widget.clear()
            for action in self.framework.data_manager.get_actions(data_type):
                list_widget.addItem(action)
                
    def on_tab_changed(self, index):
        self.cleanup_animation()
        tab_name = self.tabs.tabText(index)
        self.update_visualization()
        self.search_widget.search_field.clear()
        self.search_widget.hide_dropdown()
        
    def update_sample_spinner(self):
        current_tab = self.tabs.currentWidget()
        if current_tab:
            data_type = self.tabs.tabText(self.tabs.currentIndex()).lower().replace(' ', '_')
            action = self.action_combo.currentText()
            sample_count = self.framework.data_manager.get_sample_count(data_type, action)
            self.sample_spinner.setRange(0, max(0, sample_count - 1))

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
        self.update_log("Loading and visualizing data...")
        if not self.framework.actions:
            self.update_log("No data available for visualization.")
            return

        self.action_combo.clear()
        self.action_combo.addItems(sorted(self.framework.actions))

        max_samples = max(self.framework.data_manager.get_sample_count(data_type, action)
                          for data_type in ['raw_data', 'cleaned_data', 'validation_data']
                          for action in self.framework.data_manager.get_actions(data_type))

        if max_samples > 0:
            self.sample_spinner.setRange(0, max_samples - 1)
        else:
            self.sample_spinner.setRange(0, 0)
            self.update_log("No samples found in the dataset.")
            return

        if self.framework.data_manager.get_actions('cleaned_data'):
            self.tabs.setCurrentIndex(1)  # Switch to "Cleaned Data" tab
        elif self.framework.data_manager.get_actions('raw_data'):
            self.tabs.setCurrentIndex(0)  # Switch to "Raw Data" tab
        else:
            self.update_log("No data available for visualization in any category.")

        self.update_visualization()

    def visualize_data_for_tab(self, data_type):
        action = self.action_combo.currentText()
        sample_index = self.sample_spinner.value()

        data = self.framework.data_manager.load_data(data_type, action, sample_index)
        if data is None:
            self.update_log(f"Failed to load data for {action}, sample {sample_index}")
            self.clear_visualizations()
            return

        self.update_log(f"Visualizing {data_type} for {action}, sample {sample_index}")
        self.plot_2d.plot_eeg(data, action, sample_index, data_type, 
                              self.selected_channels, self.selected_frequencies)
        self.update_3d_animation(data, action, sample_index, data_type)
        self.update_explanations(data, action)
        
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
        if data is None or data.ndim != 3:
            self.update_log(f"Invalid data format for 3D animation: {data_type} data.")
            return

        sample = data  # The data is already the correct sample

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
        self.search_widget.search_field.clear()
        self.search_widget.hide_dropdown()

    def update_visualization(self):
        current_tab = self.tabs.currentWidget()
        if current_tab:
            data_type = self.tabs.tabText(self.tabs.currentIndex()).lower().replace(' ', '_')
            self.update_log(f"Updating visualization for {data_type}")
            self.visualize_data_for_tab(data_type)
            self.update_file_name_display()
        else:
            self.update_log("No tab selected for visualization")

    def clear_visualizations(self):
        self.cleanup_animation()
        self.plot_2d.canvas.figure.clear()
        self.plot_2d.canvas.draw()
        self.explanation_2d.clear()
        self.explanation_3d.clear()
        
    def log_and_emit(self, message):
        print(message)  # Print to console for debugging
        self.update_log(message)

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
        self.clean_data_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.framework.total_progress = 0

        worker = Worker(self.framework.clean_and_balance_data)
        worker.signals.progress.connect(self.update_log)
        worker.signals.finished.connect(self.on_data_cleaned)
        worker.signals.error.connect(self.on_worker_error)

        self.threadpool.start(worker)
        
    def on_data_cleaned(self):
        self.clean_data_btn.setEnabled(True)
        self.update_data_lists()
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "Success", "Data cleaned, balanced, and saved.")
        
        # Switch to the "cleaned_data" tab
        cleaned_data_index = self.tabs.indexOf(self.tabs.findChild(QWidget, "cleaned_data"))
        if cleaned_data_index != -1:
            self.tabs.setCurrentIndex(cleaned_data_index)
    
    def on_worker_error(self, error):
        QMessageBox.critical(self, "Error", f"An error occurred: {error}")
        self.clean_data_btn.setEnabled(True)
        self.train_data_btn.setEnabled(True)  
  
    def update_progress(self, message):
        if isinstance(message, str):
            self.update_log(message)
            if message.endswith('%'):
                try:
                    progress = int(message.split()[-1].strip('%'))
                    self.progress_bar.setValue(progress)
                except ValueError:
                    pass
        elif isinstance(message, int):
            self.progress_bar.setValue(message)

    def train_data(self):
        if not self.framework.data_manager.get_actions('cleaned_data'):
            QMessageBox.warning(self, "No Cleaned Data", "No cleaned data available. Please clean the data first.")
            return
        
        self.train_data_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.framework.total_progress = 0

        worker = Worker(self.framework.run_analysis)
        worker.signals.progress.connect(self.update_progress)
        worker.signals.result.connect(self.on_training_complete)
        worker.signals.error.connect(self.on_worker_error)

        self.threadpool.start(worker)

    def on_training_complete(self, results):
        self.train_data_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.update_log("Training complete. Check the logs for detailed results.")
        self.display_analysis_results(results)
        QMessageBox.information(self, "Success", "Training has completed.")
                
    def display_analysis_results(self, results):
        result_text = "Analysis Results:\n\n"
        for model_name, performance in results['model_performances'].items():
            result_text += f"{model_name.upper()} Model:\n"
            result_text += f"Accuracy: {performance['accuracy']:.4f}\n"
            result_text += "Classification Report:\n"
            result_text += f"{performance['classification_report']}\n\n"
        
        self.update_log(result_text)

    def update_log(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        
    def update_data_summary(self, summary):
        summary_text = "Data Summary:\n"
        for data_type, actions in summary.items():
            summary_text += f"\n{data_type.capitalize()}:\n"
            for action, details in actions.items():
                summary_text += f"  {action}: {details['count']} samples, shape: {details['shape']}\n"
        self.update_log(summary_text)

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

        if data is None or data.ndim != 3:
            print(f"Invalid data format for plotting: {data.shape if data is not None else None}")
            return

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
            channel_data = data[:, channel, :]
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = EEGAnalysisGUI()
    gui.show()
    sys.exit(app.exec_())
