import numpy as np
import pickle
import os
import tensorflow as tf
import json
import warnings
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D, Conv1D, MaxPooling1D, LSTM, Concatenate, TimeDistributed, RepeatVector, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.exceptions import ConvergenceWarning
from scipy import signal
from scipy.stats import zscore
from scipy.signal import welch
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from mne.preprocessing import ICA
from mne_connectivity import spectral_connectivity_epochs
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import keras_tuner as kt
from collections import Counter
from multiprocessing import Pool, cpu_count
from threading import Lock
import multiprocessing
import logging
from datetime import datetime
from data_manager import DataManager
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import time
import mlflow


class TemporalSpectralFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, fs=25, window_size=5, stride=2, context_size=2, overlap=0.5, max_stack_size=3, feature_version=1):
        self.fs = fs
        self.window_size = window_size
        self.stride = stride
        self.context_size = context_size
        self.overlap = overlap
        self.max_stack_size = max_stack_size
        self.feature_version = feature_version
        self.frequency_bands = {
            'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
            'beta': (13, 30), 'gamma': (30, 60)
        }
        self.adaptive_window_manager = AdaptiveWindowManager(base_window_size=window_size, 
                                                             max_stack_size=max_stack_size, 
                                                             context_size=context_size)
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for sample in X:
            sample_features = self.extract_features_windowed(sample)
            features.append(sample_features)
        return np.array(features)


    def extract_features_windowed(self, sample):
        self.debug_print(f"Input sample shape: {sample.shape}")
        windows = self.create_sliding_windows(sample)
        self.debug_print(f"Number of windows: {len(windows)}")
        self.debug_print(f"Shape of first window: {windows[0].shape}")
        features = []
        for i in range(len(windows)):
            stack_size = self.adaptive_window_manager.decide_stack_size(windows[i])
            stacked_window = self.get_stacked_window(windows, i, stack_size)
            context = self.get_context(windows, i)
            self.debug_print(f"Context shape: {np.array(context).shape}")
            window_features = self.extract_features_with_context(context, stacked_window)
            features.append(window_features)
        
        self.debug_print(f"Final features shape: {np.array(features).shape}")
        return np.array(features)

    def extract_features_with_context(self, context, stacked_window):
        self.debug_print(f"Stacked window shape: {stacked_window.shape}")
        features = self.extract_features_from_window(stacked_window)
        context_features = self.extract_context_features(context)
        return np.concatenate([features, context_features])

    def create_sliding_windows(self, data):
        return [data[i:i+self.window_size] for i in range(0, len(data) - self.window_size + 1, self.stride)]

    def extract_features_from_window(self, window):
        self.debug_print(f"Window shape in extract_features_from_window: {window.shape}")
        features = []
        
        # Spectral power features
        for band, (low, high) in self.frequency_bands.items():
            band_power = self.compute_band_power(window, low, high)
            features.extend([np.mean(band_power), np.std(band_power), np.max(band_power), np.min(band_power),
                             np.median(band_power), np.ptp(band_power), np.mean(np.diff(band_power))])

        # Compute ratios
        delta = self.compute_band_power(window, *self.frequency_bands['delta'])
        theta = self.compute_band_power(window, *self.frequency_bands['theta'])
        alpha = self.compute_band_power(window, *self.frequency_bands['alpha'])
        beta = self.compute_band_power(window, *self.frequency_bands['beta'])
        gamma = self.compute_band_power(window, *self.frequency_bands['gamma'])

        ratio_features = [
            np.mean(beta / (alpha + theta)),
            np.mean((beta + gamma) / (delta + theta)),
            np.mean(theta / alpha),
        ]
        features.extend(ratio_features)

        # Compute temporal stability
        for band in [delta, theta, alpha, beta, gamma]:
            features.append(np.mean(np.abs(np.diff(band))))

        # Spectral entropy (using pre-computed power)
        spectral_entropy = -np.sum(window * np.log2(window + 1e-10), axis=2).mean(axis=1)
        features.extend(spectral_entropy)

        # Time-domain features (adapted for frequency domain data)
        hjorth_activity = np.var(window, axis=(0, 2)).mean()
        hjorth_mobility = np.sqrt(np.var(np.diff(window, axis=0), axis=(0, 2)) / np.var(window, axis=(0, 2))).mean()
        hjorth_complexity = (np.sqrt(np.var(np.diff(np.diff(window, axis=0), axis=0), axis=(0, 2)) / 
                             np.var(np.diff(window, axis=0), axis=(0, 2))) / hjorth_mobility).mean()
        features.extend([hjorth_activity, hjorth_mobility, hjorth_complexity])

        # Wavelet features (adapted for frequency domain data)
        wavelet_features = [np.mean(np.abs(window[:, :, i:i+8])) for i in range(0, window.shape[2], 8)]
        features.extend(wavelet_features)

        # Commitment feature
        commitment = np.mean(beta) / (np.mean(alpha) + np.mean(theta))
        features.append(commitment)

        # Connectivity features
        conn_features = self.compute_connectivity_features(window)
        features.extend(conn_features)

        return np.array(features)


    def compute_band_power(self, data, low_freq, high_freq):
        freq_bins = np.arange(60)  # 0-59 Hz
        idx = np.logical_and(freq_bins >= low_freq, freq_bins < high_freq)
        return np.mean(data[:, :, idx], axis=2)

    def compute_connectivity_features(self, data):
        # Compute connectivity across channels for each frequency bin
        n_channels = data.shape[1]
        n_freq_bins = data.shape[2]
        connectivity = np.zeros((n_channels, n_channels, n_freq_bins))
        
        for f in range(n_freq_bins):
            freq_data = data[:, :, f]
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    connectivity[i, j, f] = np.abs(np.corrcoef(freq_data[:, i], freq_data[:, j])[0, 1])
                    connectivity[j, i, f] = connectivity[i, j, f]
        
        # Compute summary statistics
        mean_connectivity = np.mean(connectivity, axis=(0, 1))
        max_connectivity = np.max(connectivity, axis=(0, 1))
        std_connectivity = np.std(connectivity, axis=(0, 1))
        
        return np.concatenate([mean_connectivity, max_connectivity, std_connectivity])
    
    def debug_print(self, message):
        print(f"[DEBUG] {message}")
        
class IntentCommitmentAnalyzer:
    def __init__(self, time_threshold=1.0, commitment_threshold=0.6):
        self.time_threshold = time_threshold
        self.commitment_threshold = commitment_threshold

    def analyze(self, predictions, commitments, timestamps):
        intents = []
        current_intent = None
        intent_start = None
        
        for i, (pred, comm, time) in enumerate(zip(predictions, commitments, timestamps)):
            if comm > self.commitment_threshold:
                if current_intent is None:
                    current_intent = pred
                    intent_start = time
                elif pred != current_intent:
                    if time - intent_start >= self.time_threshold:
                        intents.append((current_intent, intent_start, time))
                    current_intent = pred
                    intent_start = time
            else:
                if current_intent is not None:
                    if time - intent_start >= self.time_threshold:
                        intents.append((current_intent, intent_start, time))
                    current_intent = None
                    intent_start = None
        
        return intents
    
class AdaptiveWindowManager:
    def __init__(self, base_window_size=5, max_stack_size=3, learning_rate=0.01):
        self.base_window_size = base_window_size
        self.max_stack_size = max_stack_size
        self.learning_rate = learning_rate
        self.stack_decision_model = self.build_stack_decision_model()
        self.online_learning_rate = learning_rate * 0.1

    def build_stack_decision_model(self):
        eeg_input = Input(shape=(None, 16, 60))  # Variable time steps
        feature_input = Input(shape=(None,))  # Variable number of features
        performance_input = Input(shape=(1,))
        
        x = Conv2D(32, (3, 3), activation='relu')(eeg_input)
        x = GlobalAveragePooling2D()(x)
        
        feature_x = Dense(32, activation='relu')(feature_input)
        feature_x = GlobalAveragePooling1D()(feature_x)
        
        combined = Concatenate()([x, feature_x, performance_input])
        x = Dense(64, activation='relu')(combined)
        outputs = Dense(self.max_stack_size, activation='softmax')(x)
        
        model = Model(inputs=[eeg_input, feature_input, performance_input], outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def decide_stack_size(self, window, features, current_performance):
        prediction = self.stack_decision_model.predict([
            window[np.newaxis, ...], 
            features[np.newaxis, ...], 
            np.array([[current_performance]])
        ])
        return np.argmax(prediction[0]) + 1

    def update(self, window, features, chosen_stack, performance):
        target = np.zeros((1, self.max_stack_size))
        target[0, chosen_stack - 1] = 1
        
        # Calculate performance improvement
        if chosen_stack > 1:
            prev_prediction = self.stack_decision_model.predict([
                window[np.newaxis, ...], 
                features[np.newaxis, ...], 
                np.array([[0]])  # Assume no previous performance
            ])
            prev_stack = np.argmax(prev_prediction[0]) + 1
            improvement = performance - prev_prediction[0, prev_stack - 1]
        else:
            improvement = performance
        
        # Train the model with the improvement as the performance input
        self.stack_decision_model.train_on_batch(
            [window[np.newaxis, ...], features[np.newaxis, ...], np.array([[improvement]])], 
            target
        )
        
    def update_online(self, window, features, chosen_stack, performance):
        target = np.zeros((1, self.max_stack_size))
        target[0, chosen_stack - 1] = 1
        
        # Use a custom training step with a lower learning rate
        with tf.GradientTape() as tape:
            predictions = self.stack_decision_model([window[np.newaxis, ...], 
                                                     features[np.newaxis, ...], 
                                                     np.array([[performance]])])
            loss = tf.keras.losses.categorical_crossentropy(target, predictions)
        
        gradients = tape.gradient(loss, self.stack_decision_model.trainable_variables)
        for grad, var in zip(gradients, self.stack_decision_model.trainable_variables):
            var.assign_sub(self.online_learning_rate * grad)
        
#         # Update stack preferences based on overall performance
#         accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(true_labels, axis=1))
#         self.stack_preferences += self.learning_rate * (accuracy - 0.5) * (self.stack_preferences - np.mean(self.stack_preferences))
#         
#         # Normalize preferences
#         self.stack_preferences = np.clip(self.stack_preferences, 0.1, None)
#         self.stack_preferences /= self.stack_preferences.sum()

#     def get_stacked_window(self, data, start_index):
#         stack_size = self.decide_stack_size(data[start_index:start_index+self.base_window_size])
#         end_index = min(start_index + stack_size * self.base_window_size, len(data))
#         return data[start_index:end_index]
# 
#     def get_context(self, data, center_index):
#         start = max(0, center_index - self.context_size * self.base_window_size)
#         end = min(len(data), center_index + (self.context_size + 1) * self.base_window_size)
#         return data[start:end]
    
class EEGAnalysis:
    def __init__(self, base_dir, fs=25):
        self.base_dir = base_dir
        self.data_manager = DataManager(base_dir)
        self.logs_dir = os.path.join(base_dir, "logs")
        self.models_dir = os.path.join(base_dir, "models")
        self.fs = fs
        self.actions = set()
        self.models = {}
        self.frequency_bands = {
            'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
            'beta': (13, 30), 'gamma': (30, 60)
        }
        self.setup_logging()
        self.progress_queue = Queue()
        self.progress_lock = Lock()
        self.total_progress = 0
        self.last_progress_update = 0
        self.num_threads = multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.window_size = 5  # 200ms at 25Hz
        self.stride = 2  # 80ms at 25Hz
        self.context_size = 2
        self.overlap = 0.5
        self.max_stack_size = 3
        self.batch_size = 32
        self.feature_version = 1
        self.stack_specific_cnn_lstm_models = {}
        self.stack_specific_unsupervised_models = {}
        self.metadata = {
            'window_size': self.window_size,
            'stride': self.stride,
            'context_size': self.context_size,
            'overlap': self.overlap,
            'max_stack_size': self.max_stack_size,
            'feature_version': self.feature_version
        }
        self.adaptive_window_manager = AdaptiveWindowManager(base_window_size=self.window_size, 
                                                             max_stack_size=self.max_stack_size, 
                                                             context_size=self.context_size)
        self.ts_extractor = TemporalSpectralFeatureExtractor(**self.metadata)
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("EEG_Analysis")
        
    def update_progress(self, progress, message=""):
        with self.progress_lock:
            self.total_progress = progress
            self.progress_queue.put((progress, message))
                
    def force_progress_complete(self, message="Task completed"):
        with self.progress_lock:
            self.total_progress = 100
            self.last_progress_update = 100
            self.progress_queue.put((100, message))

            
    def setup_logging(self):
        os.makedirs(self.logs_dir, exist_ok=True)
        log_file = os.path.join(self.logs_dir, f'eeg_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def log_and_emit(self, message, level=logging.INFO):
        self.logger.log(level, message)
        self.update_progress(0, message)  # Update progress with message but no increment
        return message
    
    def save_metadata(self):
        metadata_path = os.path.join(self.base_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)

    def load_metadata(self):
        metadata_path = os.path.join(self.base_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                # Update class attributes from metadata
                for key, value in self.metadata.items():
                    setattr(self, key, value)

    def load_data(self, progress_callback=None):
        self.actions = set()
        total_actions = sum(len(self.data_manager.get_actions(data_type)) for data_type in self.data_manager.data_catalog.keys())
        processed_actions = 0
        for data_type in self.data_manager.data_catalog.keys():
            for action in self.data_manager.get_actions(data_type):
                self.actions.add(action)
                processed_actions += 1
                progress = int((processed_actions / total_actions) * 100)
                self.update_progress(progress, f"Loading {data_type}... {progress}%")
        return self.log_data_summary()

    def log_data_summary(self):
        summary = self.data_manager.get_data_summary()
        self.logger.info(f"Loaded actions: {self.actions}")
        self.logger.info("Data summary:")
        for data_type, actions in summary.items():
            self.logger.info(f"  {data_type}:")
            for action, details in actions.items():
                self.logger.info(f"    {action}: {details['count']} samples, shape: {details['shape']}")
        return summary

    def clean_and_balance_data(self, progress_callback=None):
        if not self.actions or not self.data_manager.get_actions('raw_data'):
            return self.log_and_emit("No raw data available to clean and balance.")

        # First pass: Count valid samples and store valid sample names
        valid_sample_counts = {}
        valid_sample_names = {}
        for action in self.actions:
            action_dir = os.path.join(self.data_manager.data_dir, "raw_data", action)
            if os.path.exists(action_dir):
                valid_samples = []
                for sample_name in os.listdir(action_dir):
                    if sample_name.endswith('.npy'):
                        sample_path = os.path.join(action_dir, sample_name)
                        try:
                            sample = np.load(sample_path)
                            if sample.shape == (250, 16, 60):
                                valid_samples.append(sample_name)
                        except Exception as e:
                            self.log_and_emit(f"Error processing {sample_name} for action {action}: {str(e)}")
                valid_sample_counts[action] = len(valid_samples)
                valid_sample_names[action] = valid_samples

        # Determine the minimum number of valid samples across all actions
        min_samples = min(valid_sample_counts.values())
        if min_samples == 0:
            return self.log_and_emit("No valid data available for cleaning and balancing.")

        self.log_and_emit(f"Minimum number of valid samples across all actions: {min_samples}")

        # Second pass: Save balanced data
        total_samples_to_process = min_samples * len(self.actions)
        processed_samples = 0

        for action in self.actions:
            np.random.shuffle(valid_sample_names[action])
            for i, sample_name in enumerate(valid_sample_names[action][:min_samples]):
                sample_path = os.path.join(self.data_manager.data_dir, "raw_data", action, sample_name)
                sample = np.load(sample_path)
                self.data_manager.save_data('cleaned_data', action, sample, sample_name)
                processed_samples += 1
                progress = int((processed_samples / total_samples_to_process) * 100)
                self.update_progress(progress, f"Cleaning and balancing data... {progress}%")

        # Verify the final counts
        final_counts = {action: self.data_manager.get_sample_count('cleaned_data', action) for action in self.actions}
        if len(set(final_counts.values())) != 1:
            self.log_and_emit(f"Warning: Not all actions have the same number of samples after balancing. Counts: {final_counts}")

        self.update_progress(100, "Data cleaning and balancing completed")
        return self.log_and_emit(f"Data cleaning and balancing completed. Each action now has {min_samples} samples.")
    
    def diff_summaries(self, initial, final):
        diff = {}
        for data_type in initial.keys():
            diff[data_type] = {}
            for action in set(initial[data_type].keys()) | set(final[data_type].keys()):
                if action in initial[data_type] and action in final[data_type]:
                    if initial[data_type][action] != final[data_type][action]:
                        diff[data_type][action] = {
                            'before': initial[data_type][action],
                            'after': final[data_type][action]
                        }
                elif action in initial[data_type]:
                    diff[data_type][action] = {'before': initial[data_type][action], 'after': 'Removed'}
                else:
                    diff[data_type][action] = {'before': 'Not present', 'after': final[data_type][action]}
        return diff
    
#     def update_rf_model(self, train_X, train_y):
#         rf = self.models['rf']
#         rf.fit(train_X.reshape(train_X.shape[0], -1), np.argmax(train_y, axis=1))
#         joblib.dump(rf, os.path.join(self.models_dir, 'rf_model.joblib'))
# 
#     def train_rf_model(self, train_X, train_y):
#         return self.optimize_rf_model(train_X, train_y)
# 
#     def optimize_rf_model(self, train_data, val_data):
#         param_grid = {
#             'n_estimators': [100, 200, 300],
#             'max_depth': [10, 20, 30, None],
#             'min_samples_split': [2, 5, 10],
#             'min_samples_leaf': [1, 2, 4]
#         }
#         rf = RandomForestClassifier(random_state=42)
#         
#         # Prepare data for GridSearchCV
#         X_train, y_train = [], []
#         for batch_X, batch_y in train_data:
#             X_train.extend(batch_X.reshape(batch_X.shape[0], -1))
#             y_train.extend(np.argmax(batch_y, axis=1))
#         
#         X_train = np.array(X_train)
#         y_train = np.array(y_train)
#         
#         grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
#         grid_search.fit(X_train, y_train)
#         
#         self.log_and_emit(f"Best parameters found: {grid_search.best_params_}")
#         self.log_and_emit(f"Best cross-validation score: {grid_search.best_score_:.4f}")
#         
#         self.models['optimized_rf'] = grid_search.best_estimator_
#         return self.models['optimized_rf']

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.models_dir, 'training_history.png')
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        
        plt.show()

    def evaluate_model(self, model_type, test_data):
        y_pred = []
        y_true = []
        
        for batch_data, batch_labels in test_data:
            processed_data = self.process_data_with_adaptive_windows(batch_data)
            batch_pred = self.predict_with_stack_specific_models(processed_data)
            y_pred.extend(batch_pred)
            y_true.extend(batch_labels)
        
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        accuracy = accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        conf_matrix = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        class_report = classification_report(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), target_names=self.actions)

        self.log_and_emit(f"{model_type.upper()} Model Evaluation:")
        self.log_and_emit(f"Accuracy: {accuracy:.4f}")
        self.log_and_emit("\nConfusion Matrix:")
        self.log_and_emit(conf_matrix)
        self.log_and_emit("\nClassification Report:")
        self.log_and_emit(class_report)

        self.plot_confusion_matrix(conf_matrix, self.actions, title=f"{model_type.upper()} Model Confusion Matrix")

        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }

    def evaluate_unsupervised_models(self, stack_size, val_data):
        unsupervised_models = self.stack_specific_unsupervised_models[stack_size]
        
        autoencoder_loss = 0
        recurrent_loss = 0
        dec_loss = 0
        som_quantization_error = 0
        num_samples = 0
        
        for batch_data, _ in val_data:
            processed_data = self.process_data_with_adaptive_windows(batch_data)
            stack_data = [sample for sample in processed_data if sample[2] == stack_size]
            
            for stacked_window, _, _, _ in stack_data:
                autoencoder_loss += unsupervised_models['autoencoder'][0].test_on_batch(stacked_window[np.newaxis, ...], stacked_window[np.newaxis, ...])
                recurrent_loss += unsupervised_models['recurrent_autoencoder'].test_on_batch(stacked_window[np.newaxis, ...], stacked_window[np.newaxis, ...])
                dec_loss += unsupervised_models['dec_model'].test_on_batch(stacked_window[np.newaxis, ...], None)
                som_quantization_error += unsupervised_models['som'].quantization_error(stacked_window.flatten())
                num_samples += 1
        
        self.log_and_emit(f"Stack size {stack_size} unsupervised model evaluation:")
        self.log_and_emit(f"Autoencoder loss: {autoencoder_loss / num_samples}")
        self.log_and_emit(f"Recurrent autoencoder loss: {recurrent_loss / num_samples}")
        self.log_and_emit(f"DEC model loss: {dec_loss / num_samples}")
        self.log_and_emit(f"SOM quantization error: {som_quantization_error / num_samples}")

    def plot_confusion_matrix(self, cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    def get_feature_shape(self):
        # Get the shape of a single feature set
        for action in self.data_manager.get_actions('features'):
            feature = self.data_manager.load_data('features', action, 0)
            if feature is not None:
                return feature.shape
        raise ValueError("No features found to determine shape")

    def build_stack_specific_models(self):
        for stack_size in range(1, self.max_stack_size + 1):
            self.stack_specific_unsupervised_models[stack_size] = {
                'autoencoder': self.build_autoencoder(stack_size),
                'dec_model': self.build_dec_model(stack_size),
                'recurrent_autoencoder': self.build_recurrent_autoencoder(stack_size),
                'som': MiniSom(10, 10, self.window_size * 16 * 60 * stack_size, sigma=1.0, learning_rate=0.5)
            }
            self.stack_specific_cnn_lstm_models[stack_size] = self.build_tunable_cnn_lstm_model(stack_size)

    def build_autoencoder(self, stack_size):
        input_shape = (self.window_size, 16, 60 * stack_size)
        inputs = Input(shape=input_shape)
        
        # Encoder
        x = TimeDistributed(Dense(32, activation='relu'))(inputs)
        x = LSTM(64, return_sequences=True)(x)
        encoded = LSTM(32, return_sequences=False)(x)
        
        # Decoder
        x = Dense(64, activation='relu')(encoded)
        x = Dense(128, activation='relu')(x)
        decoded = Dense(self.window_size * 16 * 60, activation='linear')(x)
        decoded = Reshape((self.window_size, 16, 60))(decoded)
        
        autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)
        
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder, encoder
    
    def build_dec_model(self, stack_size):
        input_shape = (self.window_size, 16, 60 * stack_size)
        inputs = Input(shape=input_shape)
        x = TimeDistributed(Dense(64, activation='relu'))(inputs)
        x = LSTM(32, return_sequences=False)(x)
        clustering_layer = ClusteringLayer(n_clusters=5)(x)  # Adjust n_clusters as needed
        
        model = Model(inputs=inputs, outputs=clustering_layer)
        model.compile(optimizer='adam', loss='kld')
        
        return model

    def build_recurrent_autoencoder(self, stack_size):
        input_shape = (self.window_size, 16, 60 * stack_size)
        inputs = Input(shape=input_shape)
        
        # Encoder
        encoded = LSTM(64, return_sequences=True)(inputs)
        encoded = LSTM(32, return_sequences=False)(encoded)
        
        # Decoder
        decoded = RepeatVector(self.window_size)(encoded)
        decoded = LSTM(32, return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(16 * 60))(decoded)
        decoded = Reshape((self.window_size, 16, 60))(decoded)
        
        model = Model(inputs, decoded)
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def build_tunable_cnn_lstm_model(self, stack_size):
        def create_model(hp):
            input_shape = (None, self.window_size, 16, 60 * stack_size)
            num_classes = len(self.actions)

            # Use the same model architecture as in optimize_combined_model
            input_layer = Input(shape=(None, input_shape[1]))
            x_cnn = Conv1D(hp.Int('cnn_filters_1', 32, 128, step=32), 
                           hp.Int('cnn_kernel_1', 3, 9, step=2), 
                           activation='relu')(input_layer)
            x_cnn = MaxPooling1D(2)(x_cnn)
            x_cnn = Conv1D(hp.Int('cnn_filters_2', 64, 256, step=64), 
                           hp.Int('cnn_kernel_2', 3, 9, step=2), 
                           activation='relu')(x_cnn)
            x_cnn = MaxPooling1D(2)(x_cnn)
            
            temp_attention = Attention()([x_cnn, x_cnn])
            x_cnn = Concatenate()([x_cnn, temp_attention])
            
            x_cnn = LSTM(hp.Int('lstm_units_1', 32, 128, step=32), return_sequences=True)(x_cnn)
            x_cnn = LSTM(hp.Int('lstm_units_2', 16, 64, step=16))(x_cnn)
            
            x_transformer = input_layer
            for _ in range(hp.Int('num_transformer_blocks', 1, 3)):
                x_transformer = MultiHeadAttention(num_heads=hp.Int('num_heads', 2, 8),
                                                   key_dim=hp.Int('key_dim', 16, 64, step=16))(x_transformer, x_transformer)
                x_transformer = LayerNormalization(epsilon=1e-6)(x_transformer)
                x_transformer = Conv1D(filters=hp.Int('transformer_conv_filters', 16, 64, step=16),
                                       kernel_size=1, activation="relu")(x_transformer)
                x_transformer = LayerNormalization(epsilon=1e-6)(x_transformer)
            x_transformer = GlobalAveragePooling1D()(x_transformer)
            
            combined = Concatenate()([x_cnn, x_transformer])
            combined = Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu')(combined)
            output = Dense(num_classes, activation='softmax')(combined)
            
            model = Model(inputs=input_layer, outputs=output)
            model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            return model
        
        return kt.Hyperband(
            create_model,
            objective='val_accuracy',
            max_epochs=50,
            factor=3,
            directory=f'cnn_lstm_tuning_stack_{stack_size}',
            project_name=f'EEG_cnn_lstm_tuning_stack_{stack_size}'
        )
    
    def evaluate_epoch(self):
        val_loss = 0
        val_acc = 0
        num_batches = 0
        
        for batch_data, batch_labels in self.raw_data_generator('validation_data'):
            processed_data = self.process_data_with_adaptive_windows(batch_data)
            
            batch_loss = 0
            batch_acc = 0
            for stack_size in range(1, self.max_stack_size + 1):
                stack_data = [sample for sample in processed_data if sample[2] == stack_size]
                if stack_data:
                    stack_features = np.array([sample[3] for sample in stack_data])
                    stack_labels = np.array([label for label, sample in zip(batch_labels, processed_data) if sample[2] == stack_size])
                    
                    y_pred = self.stack_specific_cnn_lstm_models[stack_size].predict(stack_features)
                    
                    loss = tf.keras.losses.categorical_crossentropy(stack_labels, y_pred).numpy().mean()
                    acc = tf.keras.metrics.categorical_accuracy(stack_labels, y_pred).numpy().mean()
                    
                    batch_loss += loss
                    batch_acc += acc
            
            val_loss += batch_loss / self.max_stack_size
            val_acc += batch_acc / self.max_stack_size
            num_batches += 1
        
        return val_loss / num_batches, val_acc / num_batches


    def compute_band_power(self, eeg_data, low_freq, high_freq):
        freq_mask = np.logical_and(np.arange(eeg_data.shape[2]) >= low_freq, 
                                   np.arange(eeg_data.shape[2]) < high_freq)
        return np.mean(eeg_data[:, :, freq_mask], axis=(0, 2))
    
    def normalize_features(self, features):
        return (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    def extract_combined_features(self, stacked_window, stack_size):
        # Extract unsupervised features
        unsupervised_models = self.stack_specific_unsupervised_models[stack_size]
        autoencoder_features = unsupervised_models['autoencoder'].encoder.predict(stacked_window[np.newaxis, ...])
        dec_features = unsupervised_models['dec_model'].predict(stacked_window[np.newaxis, ...])
        recurrent_features = unsupervised_models['recurrent_autoencoder'].get_layer('encoder').predict(stacked_window[np.newaxis, ...])
        som_features = np.array([unsupervised_models['som'].winner(stacked_window.flatten())])
        
        # Combine unsupervised features
        unsupervised_features = np.concatenate([
            autoencoder_features.flatten(),
            dec_features.flatten(),
            recurrent_features.flatten(),
            som_features.flatten()
        ])
        
        # Extract hand-crafted features
        handcrafted_features = self.ts_extractor.extract_features_from_window(stacked_window)
        
        # Combine all features
        return np.concatenate([unsupervised_features, handcrafted_features])

    def raw_data_generator(self, data_type='cleaned_data', batch_size=32):
        while True:
            for action in self.actions:
                sample_count = self.data_manager.get_sample_count(data_type, action)
                for i in range(0, sample_count, batch_size):
                    batch_data = []
                    batch_labels = []
                    for j in range(i, min(i + batch_size, sample_count)):
                        data = self.data_manager.load_data(data_type, action, j)
                        if data is not None:
                            processed_data = self.process_data_with_adaptive_windows(data, is_batch=False)
                            batch_data.extend(processed_data)
                            batch_labels.extend([self.actions.index(action)] * len(processed_data))
                    
                    if batch_data:
                        yield batch_data, np.eye(len(self.actions))[batch_labels]

    def integrated_training(self, epochs=10, batch_size=32):
        self.build_stack_specific_models()
        
#         # Callbacks
#         early_stopping = 
#         model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
#             os.path.join(self.models_dir, 'integrated_optimized_model.h5'),
#             save_best_only=True, monitor='val_accuracy'
#         )

        with mlflow.start_run(run_name="integrated_training"):
            for epoch in range(epochs):
                self.log_and_emit(f"Epoch {epoch+1}/{epochs}")

                # Perform hyperparameter tuning every `tuning_frequency` epochs or at the start
                if epoch % tuning_frequency == 0:
                    for stack_size, tuner in self.stack_specific_cnn_lstm_models.items():
                        self.log_and_emit(f"Tuning CNN-LSTM model for stack size {stack_size}")
                        tuner.search(
                            self.raw_data_generator('cleaned_data', batch_size),
                            epochs=1,  # Reduce epochs for quicker tuning
                            validation_data=self.raw_data_generator('validation_data', batch_size),
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
                        )
                        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                        self.stack_specific_cnn_lstm_models[stack_size] = tuner.hypermodel.build(best_hps)

                for batch_data, batch_labels in self.raw_data_generator('cleaned_data', batch_size):
                    processed_data = self.process_data_with_adaptive_windows(batch_data)
                    
                    # Train unsupervised models
                    self.train_unsupervised_models(processed_data)
                    
                    # Train CNN-LSTM models with progressive stacking
                    stack_performances = self.train_cnn_lstm_models_progressive(processed_data, batch_labels)
                    
                    # Update unsupervised models' weights based on CNN-LSTM performance
                    for sample_performance in stack_performances:
                        for stack_size, (accuracy, stacked_window, features) in sample_performance:
                            unsupervised_models = self.stack_specific_unsupervised_models[stack_size]
                            weight_update = accuracy * self.learning_rate
                            for model in [unsupervised_models['autoencoder'][0], unsupervised_models['recurrent_autoencoder'], unsupervised_models['dec_model']]:
                                for layer in model.layers:
                                    if hasattr(layer, 'kernel'):
                                        layer.kernel.assign_add(layer.kernel * weight_update)
                                    if hasattr(layer, 'bias'):
                                        layer.bias.assign_add(layer.bias * weight_update)

                    # Update AdaptiveWindowManager
                    for sample_performance in stack_performances:
                        current_performance = 0
                        for stack_size, (accuracy, stacked_window, features) in enumerate(sample_performance, start=1):
                            self.adaptive_window_manager.update(stacked_window, features, stack_size, accuracy)
                            if accuracy <= current_performance:
                                break
                            current_performance = accuracy
                
                val_loss, val_acc = self.evaluate_epoch()
                self.log_and_emit(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
                mlflow.log_metric(f"epoch_{epoch+1}_val_loss", val_loss)
                mlflow.log_metric(f"epoch_{epoch+1}_val_accuracy", val_acc)
                
                if (epoch + 1) % 5 == 0:
                    self.save_models(epoch + 1)
                    self.update_learning_rates()

            self.save_models('final')

        return self.stack_specific_cnn_lstm_models, self.stack_specific_unsupervised_models, self.adaptive_window_manager
    
    def process_data_with_adaptive_windows(self, data, is_batch=True):
        if is_batch:
            return [self._process_single_sample(sample) for sample in data]
        else:
            return self._process_single_sample(data)

    def _process_batch(self, batch_data):
        return [self._process_single_sample(sample) for sample in batch_data]

    def _process_single_sample(self, sample):
        windows = self.ts_extractor.create_sliding_windows(sample)
        processed_windows = []
        for i in range(len(windows)):
            stack_size = self.adaptive_window_manager.decide_stack_size(windows[i])
            stacked_window = self.get_stacked_window(sample, i, stack_size)
            context = self.get_context(sample, i)
            features = self.extract_combined_features(stacked_window, stack_size)
            processed_windows.append((stacked_window, context, stack_size, features))
        return processed_windows
    
    def get_stacked_window(self, data, start_index, stack_size):
        end_index = min(start_index + stack_size * self.window_size, len(data))
        return data[start_index:end_index]

    def get_context(self, data, center_index):
        start = max(0, center_index - self.context_size * self.window_size)
        end = min(len(data), center_index + (self.context_size + 1) * self.window_size)
        return data[start:end]
    
    def predict_with_stack_specific_models(self, batch_data):
        predictions = []
        for sample in batch_data:
            sample_predictions = []
            for _, _, stack_size, features in sample:
                model = self.stack_specific_cnn_lstm_models[stack_size]
                pred = model.predict(features[np.newaxis, ...])
                sample_predictions.append(pred[0])
            predictions.append(np.mean(sample_predictions, axis=0))
        return np.array(predictions)

    def train_unsupervised_models(self, processed_data):
        for sample in processed_data:
            for stacked_window, _, stack_size, _ in sample:
                unsupervised_models = self.stack_specific_unsupervised_models[stack_size]
                
                # Train autoencoder
                unsupervised_models['autoencoder'][0].train_on_batch(stacked_window[np.newaxis, ...], stacked_window[np.newaxis, ...])
                
                # Train recurrent autoencoder
                unsupervised_models['recurrent_autoencoder'].train_on_batch(stacked_window[np.newaxis, ...], stacked_window[np.newaxis, ...])
                
                # Train DEC model
                unsupervised_models['dec_model'].train_on_batch(stacked_window[np.newaxis, ...], None)
                
                # Train SOM
                unsupervised_models['som'].update(stacked_window.flatten(), unsupervised_models['som'].winner(stacked_window.flatten()), 0)
                    
    def train_cnn_lstm_models_progressive(self, processed_data, batch_labels):
        stack_performances = []
        for sample, label in zip(processed_data, batch_labels):
            sample_performance = []
            current_performance = 0
            for stack_size, (stacked_window, _, _, features) in enumerate(sample, start=1):
                model = self.stack_specific_cnn_lstm_models[stack_size]
                # Use both stacked_window and features
                performance = model.train_on_batch([stacked_window[np.newaxis, ...], features[np.newaxis, ...]], label[np.newaxis, ...])
                accuracy = performance[1]  # Assuming accuracy is the second metric
                
                sample_performance.append((accuracy, stacked_window, features))
                if accuracy <= current_performance:
                    break  # Stop if no improvement with larger stack
                current_performance = accuracy
            
            stack_performances.append(sample_performance)
        return stack_performances
            
    def save_models(self, identifier):
        for stack_size, model in self.stack_specific_cnn_lstm_models.items():
            model.save(os.path.join(self.models_dir, f'cnn_lstm_model_stack_{stack_size}_{identifier}.h5'))
            mlflow.keras.log_model(model, f"cnn_lstm_model_stack_{stack_size}_{identifier}")
        
        for stack_size, models in self.stack_specific_unsupervised_models.items():
            for model_name, model in models.items():
                if isinstance(model, tf.keras.Model):
                    model.save(os.path.join(self.models_dir, f'{model_name}_stack_{stack_size}_{identifier}.h5'))
                    mlflow.keras.log_model(model, f"{model_name}_stack_{stack_size}_{identifier}")
                else:
                    with open(os.path.join(self.models_dir, f'{model_name}_stack_{stack_size}_{identifier}.pkl'), 'wb') as f:
                        pickle.dump(model, f)
                    mlflow.log_artifact(os.path.join(self.models_dir, f'{model_name}_stack_{stack_size}_{identifier}.pkl'), f"{model_name}_stack_{stack_size}_{identifier}")
        
        with open(os.path.join(self.models_dir, f'adaptive_window_manager_{identifier}.pkl'), 'wb') as f:
            pickle.dump(self.adaptive_window_manager, f)
        mlflow.log_artifact(os.path.join(self.models_dir, f'adaptive_window_manager_{identifier}.pkl'), f"adaptive_window_manager_{identifier}")

    def update_learning_rates(self):
        for stack_size in range(1, self.max_stack_size + 1):
            # Update CNN-LSTM model learning rate
            cnn_lstm_model = self.stack_specific_cnn_lstm_models[stack_size]
            K.set_value(cnn_lstm_model.optimizer.learning_rate, K.get_value(cnn_lstm_model.optimizer.learning_rate) * 0.9)
            
            # Update unsupervised models learning rates
            unsupervised_models = self.stack_specific_unsupervised_models[stack_size]
            for model in [unsupervised_models['autoencoder'][0], unsupervised_models['recurrent_autoencoder'], unsupervised_models['dec_model']]:
                K.set_value(model.optimizer.learning_rate, K.get_value(model.optimizer.learning_rate) * 0.9)
            
            # Update SOM learning rate
            unsupervised_models['som'].learning_rate *= 0.9
        
        # Update AdaptiveWindowManager learning rate
        self.adaptive_window_manager.learning_rate *= 0.9

    def extract_features(self, processed_data):
        features = []
        for sample in processed_data:
            sample_features = []
            for stacked_window, context, stack_size in sample:
                unsupervised_models = self.stack_specific_unsupervised_models[stack_size]
                
                # Extract features using unsupervised models
                autoencoder_features = unsupervised_models['autoencoder'].encoder.predict(stacked_window[np.newaxis, ...])
                dec_features = unsupervised_models['dec_model'].predict(stacked_window[np.newaxis, ...])
                recurrent_features = unsupervised_models['recurrent_autoencoder'].get_layer('encoder').predict(stacked_window[np.newaxis, ...])
                som_features = np.array([unsupervised_models['som'].winner(stacked_window.flatten())])
                
                # Combine features
                combined_features = np.concatenate([
                    autoencoder_features.flatten(),
                    dec_features.flatten(),
                    recurrent_features.flatten(),
                    som_features.flatten()
                ])
                
                # Add hand-crafted features
                handcrafted_features = self.ts_extractor.extract_features_from_window(stacked_window)
                
                # Combine all features
                window_features = np.concatenate([combined_features, handcrafted_features])
                
#                 # If labels are provided, calculate feature importance
#                 if labels is not None:
#                     # Use mutual information for feature importance
#                     feature_importance = mutual_info_classif(window_features[np.newaxis, ...], labels)
#                     # Normalize feature importance
#                     feature_importance = feature_importance / np.sum(feature_importance)
#                     # Append feature importance to window features
#                     window_features = np.concatenate([window_features, feature_importance])

                sample_features.append(window_features)
            features.append(np.array(sample_features))
        
        return np.array(features)
    
    def update_feature_extraction(self):
        self.feature_version += 1
        self.data_manager.clear_cached_features(self.feature_version)
        self.log_and_emit(f"Updated feature extraction to version {self.feature_version}")

                            
    def cross_validate_models(self, X, y, n_splits=5):
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for model_name in ['combined', 'rf', 'optimized_nn']:
            if model_name == 'combined':
                model = self.create_combined_model(X.shape[1:], len(self.actions))
                scores = cross_val_score(model, [X, X], y, cv=kfold, scoring='accuracy')
            elif model_name == 'optimized_nn':
                model = self.build_tunable_model(kt.HyperParameters())
                scores = cross_val_score(model, [X, X], y, cv=kfold, scoring='accuracy')
            else:  # RF model
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                scores = cross_val_score(model, X.reshape(X.shape[0], -1), np.argmax(y, axis=1), cv=kfold, scoring='accuracy')
            
            self.log_and_emit(f"{model_name.upper()} Cross-Validation Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    def visualize_clusters(self, action):
        if action not in self.ica_components or action not in self.cluster_labels:
            self.log_and_emit(f"No data available for action: {action}")
            return

        ica_data = np.vstack([batch.reshape(-1, batch.shape[-1]) for batch in self.ica_components[action]])
        cluster_labels = np.concatenate([labels.flatten() for labels in self.cluster_labels[action]])

        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(ica_data)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f"t-SNE Visualization of EEG Clusters for {action}")
        plt.xlabel("t-SNE feature 1")
        plt.ylabel("t-SNE feature 2")
        plt.show()

    def analyze_commitment(self, eeg_data):
        alpha_power = np.mean(self.compute_band_power(eeg_data, *self.frequency_bands['alpha']))
        beta_power = np.mean(self.compute_band_power(eeg_data, *self.frequency_bands['beta']))
        theta_power = np.mean(self.compute_band_power(eeg_data, *self.frequency_bands['theta']))
        commitment = beta_power / (alpha_power + theta_power)
        return commitment
    
    def analyze_intent_and_commitment(self, predictions, commitments, timestamps):
        intents = []
        current_intent = None
        intent_start = None
        
        for i, (pred, comm, time) in enumerate(zip(predictions, commitments, timestamps)):
            if comm > self.commitment_threshold:
                if current_intent is None:
                    current_intent = pred
                    intent_start = time
                elif pred != current_intent:
                    if time - intent_start >= self.time_threshold:
                        intents.append((current_intent, intent_start, time))
                    current_intent = pred
                    intent_start = time
            else:
                if current_intent is not None:
                    if time - intent_start >= self.time_threshold:
                        intents.append((current_intent, intent_start, time))
                    current_intent = None
                    intent_start = None
        
        return intents

    def detect_intent(self, live_eeg_data, max_latency=200):
        processed_data = self.process_data_with_adaptive_windows(live_eeg_data, is_batch=False)
        predictions = []
        commitments = []
        timestamps = []
        
        for i, sample in enumerate(processed_data):
            multi_scale_predictions = []
            multi_scale_confidences = []
            current_performance = 0
            
            for stack_size, (window, _, _, features) in enumerate(sample, start=1):
                if stack_size > 1:
                    # Use AdaptiveWindowManager to decide whether to continue stacking
                    if self.adaptive_window_manager.decide_stack_size(window, features, current_performance) < stack_size:
                        break

                model = self.stack_specific_cnn_lstm_models[stack_size]
                pred = model.predict(features[np.newaxis, ...])[0]
                confidence = np.max(pred)
                
                if confidence > current_performance:
                    current_performance = confidence
                    multi_scale_predictions.append(pred)
                    multi_scale_confidences.append(confidence)
                else:
                    break  # Stop if no improvement with larger stack
            
            # Choose the prediction with the highest confidence
            best_scale = np.argmax(multi_scale_confidences)
            predicted_action = self.actions[np.argmax(multi_scale_predictions[best_scale])]
            commitment = self.analyze_commitment(sample[0][0])  # Use the first window for commitment analysis
            
            predictions.append(predicted_action)
            commitments.append(commitment)
            timestamps.append(i * self.stride / self.fs)
            
            # Check if we've exceeded the maximum latency
            if timestamps[-1] * 1000 > max_latency:
                break

        return predictions, commitments, timestamps

    def visualize_intent_and_commitment(self, predictions, commitments, timestamps, intents):
        plt.figure(figsize=(12, 8))
        
        # Plot predicted actions
        plt.subplot(3, 1, 1)
        action_indices = [self.actions.index(action) for action in predictions]
        plt.plot(timestamps, action_indices, marker='o')
        plt.yticks(range(len(self.actions)), self.actions)
        plt.title('Predicted Actions Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Predicted Action')
        
        # Plot commitment levels
        plt.subplot(3, 1, 2)
        plt.plot(timestamps, commitments, marker='o')
        plt.title('Commitment Levels Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Commitment Level')
        
        # Plot detected intents
        plt.subplot(3, 1, 3)
        for intent in intents:
            action, start_time, end_time = intent
            plt.axvspan(start_time, end_time, alpha=0.3, label=action)
        plt.title('Detected Intents')
        plt.xlabel('Time (s)')
        plt.ylabel('Intent')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def run_analysis(self):
        try:
            self.log_and_emit("Preparing data for model training...")
            train_data = self.raw_data_generator('cleaned_data', self.batch_size)
            val_data = self.raw_data_generator('validation_data', self.batch_size)

            self.log_and_emit("Training and optimizing models...")
            self.integrated_training(epochs=10, batch_size=self.batch_size)

            self.log_and_emit("Evaluating models...")
            for stack_size in range(1, self.max_stack_size + 1):
                self.log_and_emit(f"Evaluating CNN-LSTM model for stack size {stack_size}")
                self.evaluate_model(f'cnn_lstm_stack_{stack_size}', val_data)
                
                self.log_and_emit(f"Evaluating unsupervised models for stack size {stack_size}")
                self.evaluate_unsupervised_models(stack_size, val_data)

            self.log_and_emit("Analyzing intents and commitments...")
            sample_val_data = next(val_data)[0]  # Get the first batch of validation data
            predictions, commitments, timestamps = self.detect_intent(sample_val_data[0])  # Analyze first sample in batch
            intents = self.analyze_intent_and_commitment(predictions, commitments, timestamps)

            self.log_and_emit("Visualizing results...")
            self.visualize_intent_and_commitment(predictions, commitments, timestamps, intents)

            self.log_and_emit("Analysis complete.")
        except Exception as e:
            self.log_and_emit(f"Error during analysis: {str(e)}", level=logging.ERROR)
            raise
    
#     def analyze_feature_importance(self, X, y):
#         if 'optimized_rf' in self.models:
#             rf_model = self.models['optimized_rf']
#             importances = rf_model.feature_importances_
#             feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
#             importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
#             importance_df = importance_df.sort_values('importance', ascending=False)
#             
#             plt.figure(figsize=(10, 6))
#             sns.barplot(x='importance', y='feature', data=importance_df.head(20))
#             plt.title('Top 20 Most Important Features')
#             plt.tight_layout()
#             plt.show()
#         else:
#             print("Random Forest model not found. Please train the model first.")

