import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D, Conv1D, MaxPooling1D, LSTM, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
from PyQt5.QtCore import QObject, pyqtSignal
import logging
import os
from datetime import datetime

class EEGAnalysis(QObject):
    progress_update = pyqtSignal(str)
    
    def __init__(self, base_dir, fs=250, lowcut=1, highcut=60, n_ica_components=16, n_clusters=5):
        super().__init__()
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "model_data")
        self.logs_dir = os.path.join(base_dir, "logs")
        self.models_dir = os.path.join(base_dir, "models")
        self.ensure_directories()
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.n_ica_components = n_ica_components
        self.n_clusters = n_clusters
        self.data = {
            'raw_data': {},
            'cleaned_data': {},
            'preprocessed_data': {},
            'validation_data': {}
        }
        self.preprocessed_data = {}
        self.preprocessing_settings = {}
        self.ica_components = {}
        self.cluster_labels = {}
        self.actions = set()
        self.models = {}
        self.frequency_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 60)
        }
        
        self.ensure_directories()
        
        # Set up logging
        os.makedirs(self.logs_dir, exist_ok=True)
        log_file = os.path.join(self.logs_dir, f'eeg_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def ensure_directories(self):
        for dir_path in [self.data_dir, self.logs_dir, self.models_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        for subdir in ["raw_data", "cleaned_data", "preprocessed_data", "validation_data"]:
            os.makedirs(os.path.join(self.data_dir, subdir), exist_ok=True)

    def log_and_emit(self, message, level=logging.INFO):
        self.logger.log(level, message)
        self.progress_update.emit(message)
        
    def get_data_for_visualization(self, data_type):
        if data_type not in self.data:
            self.log_and_emit(f"Invalid data type: {data_type}")
            return None

        if not self.data[data_type]:
            self.log_and_emit(f"No data available for {data_type}")
            return None

        return self.data[data_type]

    def load_data(self):
        self.data = {key: {} for key in self.data.keys()}
        self.actions = set()

        for data_type in self.data.keys():
            data_type_dir = os.path.join(self.data_dir, data_type)
            if not os.path.exists(data_type_dir):
                self.log_and_emit(f"Directory not found: {data_type_dir}")
                continue

            for action in os.listdir(data_type_dir):
                action_dir = os.path.join(data_type_dir, action)
                if os.path.isdir(action_dir):
                    self.actions.add(action)
                    self.data[data_type][action] = []
                    for file in os.listdir(action_dir):
                        if file.endswith('.npy'):
                            try:
                                eeg_data = np.load(os.path.join(action_dir, file))
                                if eeg_data.shape[1:] == (16, 60):
                                    self.data[data_type][action].append(eeg_data)
                                else:
                                    self.log_and_emit(f"Skipping file with unexpected shape: {file}")
                            except Exception as e:
                                self.log_and_emit(f"Error loading file {file}: {str(e)}")

        self.log_and_emit(f"Loaded actions: {self.actions}")
        self.log_and_emit(f"Data summary:")
        for data_type, actions in self.data.items():
            self.log_and_emit(f"  {data_type}:")
            for action, data_list in actions.items():
                self.log_and_emit(f"    {action}: {len(data_list)} samples")

    def preprocess_all_data(self):
        for action, batches in self.data.items():
            self.preprocessed_data[action] = [self.preprocess_eeg(batch, action) for batch in batches]
            
    def validate_data(self, data):
        if np.isnan(data).any():
            self.log_and_emit("Warning: NaN values detected in the data. Replacing with zeros.")
            data = np.nan_to_num(data)
        if np.isinf(data).any():
            self.log_and_emit("Warning: Infinite values detected in the data. Replacing with large finite values.")
            data = np.clip(data, -1e15, 1e15)
        return data
    
    def set_preprocessing_settings(self, action, settings):
        self.preprocessing_settings[action] = settings

    def preprocess_eeg(self, data, action):
        settings = self.preprocessing_settings.get(action, {})
        
        # Bandpass filtering
        lowcut = settings.get('lowcut', 1)
        highcut = settings.get('highcut', 60)
        filtered_data = self.apply_bandpass_filter(data, lowcut, highcut)
        
        # Notch filtering
        notch_freq = settings.get('notch_freq', 50.0)
        quality_factor = settings.get('quality_factor', 30.0)
        notched_data = self.apply_notch_filter(filtered_data, notch_freq, quality_factor)
        
        # Adaptive filtering for artifact removal
        mu = settings.get('adaptive_mu', 0.01)
        order = settings.get('adaptive_order', 5)
        adaptive_filtered_data = self.apply_adaptive_filter(notched_data, mu, order)
        
        # Artifact removal using FastICA
        n_components = settings.get('ica_components', data.shape[2])
        ica_data = self.apply_ica(adaptive_filtered_data, n_components)
        
        # Normalization
        normalized_data = zscore(ica_data, axis=0)
        
        return normalized_data
    
    def apply_bandpass_filter(self, data, lowcut, highcut):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i, :] = signal.filtfilt(b, a, data[:, i, :], axis=0)
        return filtered_data

    def apply_notch_filter(self, data, notch_freq, quality_factor):
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, self.fs)
        notched_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            notched_data[:, i, :] = signal.filtfilt(b_notch, a_notch, data[:, i, :], axis=0)
        return notched_data

    def apply_adaptive_filter(self, data, mu=0.01, order=5):
        filtered_data = np.zeros_like(data)
        for channel in range(data.shape[1]):
            for time_point in range(data.shape[2]):
                x = data[:, channel, time_point]
                d = x  # Desired signal (assuming noise is uncorrelated with the signal)
                w = np.zeros(order)
                for i in range(order, len(x)):
                    x_i = x[i-order:i][::-1]
                    y = np.dot(w, x_i)
                    e = d[i] - y
                    w_update = 2 * mu * e * x_i
                    w += np.nan_to_num(w_update, nan=0.0, posinf=1e15, neginf=-1e15)
                    filtered_data[i, channel, time_point] = y if not np.isnan(y) else x[i]
        return filtered_data

    def apply_ica(self, data, n_components):
        ica = FastICA(n_components=n_components, random_state=42, max_iter=5000, tol=1e-4)
        try:
            ica_data = ica.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
        except ValueError as e:
            self.log_and_emit(f"FastICA failed. Using original data. Error: {e}")
            ica_data = data
        return ica_data

    def cluster_ica_components(self):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        for action, batches in self.ica_components.items():
            self.cluster_labels[action] = []
            for batch in batches:
                batch_2d = batch.reshape(-1, batch.shape[-1])
                labels = kmeans.fit_predict(batch_2d)
                self.cluster_labels[action].append(labels.reshape(batch.shape[0], -1))

    def create_combined_model(self, input_shape, num_classes):
        # CNN-LSTM branch
        input_cnn = Input(shape=input_shape)
        x_cnn = Conv1D(64, 3, activation='relu')(input_cnn)
        x_cnn = MaxPooling1D(2)(x_cnn)
        x_cnn = Conv1D(128, 3, activation='relu')(x_cnn)
        x_cnn = MaxPooling1D(2)(x_cnn)
        x_cnn = LSTM(64, return_sequences=True)(x_cnn)
        x_cnn = LSTM(32)(x_cnn)
        
        # Transformer branch
        input_transformer = Input(shape=input_shape)
        x_transformer = input_transformer
        for _ in range(2):  # 2 transformer blocks
            x_transformer = MultiHeadAttention(num_heads=4, key_dim=32)(x_transformer, x_transformer)
            x_transformer = LayerNormalization(epsilon=1e-6)(x_transformer)
            x_transformer = Conv1D(filters=32, kernel_size=1, activation="relu")(x_transformer)
            x_transformer = LayerNormalization(epsilon=1e-6)(x_transformer)
        x_transformer = GlobalAveragePooling1D()(x_transformer)
        
        # Combine branches
        combined = Concatenate()([x_cnn, x_transformer])
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.5)(combined)
        output = Dense(num_classes, activation='softmax')(combined)
        
        model = Model(inputs=[input_cnn, input_transformer], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_combined_model(self, train_X, train_y, test_X, test_y, epochs=50, batch_size=32):
        model = self.create_combined_model(train_X.shape[1:], len(self.actions))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(f"{self.model_dir}/best_combined_model.h5", save_best_only=True, monitor='val_accuracy')
        
        history = model.fit([train_X, train_X], train_y, 
                            batch_size=batch_size, 
                            epochs=epochs, 
                            validation_data=([test_X, test_X], test_y),
                            callbacks=[early_stopping, model_checkpoint])
        
        self.models['combined'] = model
        self.plot_training_history(history)
        return model

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
        plt.show()

    def train_rf_model(self, train_X, train_y):
        return self.optimize_rf_model(train_X, train_y)

    def optimize_rf_model(self, X, y):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X, y)
        
        self.log_and_emit("Best parameters found: ", grid_search.best_params_)
        self.log_and_emit("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
        
        self.models['rf'] = grid_search.best_estimator_
        return self.models['rf']

    def evaluate_model(self, model_type, test_X, test_y):
        model = self.models.get(model_type)
        if model is None:
            self.log_and_emit(f"No {model_type} model found. Please train the model first.")
            return None

        if model_type in ['combined', 'optimized_nn']:
            y_pred = model.predict([test_X, test_X])
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(test_y, axis=1)
        else:  # RF model
            y_pred = model.predict(test_X.reshape(test_X.shape[0], -1))
            y_pred_classes = y_pred
            y_true_classes = np.argmax(test_y, axis=1)

        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
        class_report = classification_report(y_true_classes, y_pred_classes, target_names=self.actions)

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

    def plot_confusion_matrix(self, cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    def extract_features(self, eeg_data):
        features = []
        
        # Spectral power features
        for band, (low, high) in self.frequency_bands.items():
            band_power = self.compute_band_power(eeg_data, low, high)
            features.append(np.mean(band_power))
            features.append(np.std(band_power))
        
        # Time-domain features
        hjorth_activity = np.var(eeg_data, axis=0).mean()
        hjorth_mobility = np.sqrt(np.var(np.diff(eeg_data, axis=0), axis=0) / np.var(eeg_data, axis=0)).mean()
        hjorth_complexity = (np.sqrt(np.var(np.diff(np.diff(eeg_data, axis=0), axis=0), axis=0) / 
                             np.var(np.diff(eeg_data, axis=0), axis=0)) / hjorth_mobility).mean()
        features.extend([hjorth_activity, hjorth_mobility, hjorth_complexity])
        
        # Frequency-domain features
        freqs, psd = welch(eeg_data, fs=self.fs, nperseg=self.fs)
        spectral_entropy = -np.sum(psd * np.log2(psd), axis=1).mean()
        features.append(spectral_entropy)
        
        # Wavelet transform features
        wavelet = 'db4'
        coeffs = pywt.wavedec(eeg_data, wavelet, level=5)
        wavelet_features = [np.mean(np.abs(c)) for c in coeffs]
        features.extend(wavelet_features)
        
        # Connectivity features
        # Reshape eeg_data if necessary to match the expected input format
        eeg_epochs = eeg_data.reshape(1, *eeg_data.shape)  # Add a singleton dimension for epochs
        conn, _, _, _, _ = spectral_connectivity_epochs(eeg_epochs, method='wpli', sfreq=self.fs, fmin=1, fmax=60, n_jobs=1)
        features.extend(conn.mean(axis=2).flatten())
        
        return np.array(features)

    def compute_band_power(self, eeg_data, low_freq, high_freq):
        freq_mask = np.logical_and(np.arange(eeg_data.shape[2]) >= low_freq, 
                                   np.arange(eeg_data.shape[2]) < high_freq)
        return np.mean(eeg_data[:, :, freq_mask], axis=(0, 2))

    def prepare_data_for_training(self):
        X = []
        y = []
        for action_index, action in enumerate(self.actions):
            for batch in self.preprocessed_data[action]:
                X.append(batch)
                y.append(action_index)
        X = np.array(X)
        y = np.eye(len(self.actions))[y]  # One-hot encode the labels
        return train_test_split(X, y, test_size=0.2, random_state=42)

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

    def detect_intent(self, eeg_data, window_size=10, overlap=5):
        predictions = []
        commitments = []
        
        for i in range(0, len(eeg_data) - window_size + 1, overlap):
            window = eeg_data[i:i+window_size]
            
            # Use combined model for action prediction
            action_pred = self.models['combined'].predict([window.reshape(1, *window.shape), window.reshape(1, *window.shape)])
            predicted_action = self.actions[np.argmax(action_pred)]
            
            # Analyze commitment
            commitment = self.analyze_commitment(window)
            
            predictions.append(predicted_action)
            commitments.append(commitment)
        
        return predictions, commitments

    def visualize_intent_and_commitment(self, predictions, commitments):
        plt.figure(figsize=(12, 6))
        
        # Plot predicted actions
        plt.subplot(2, 1, 1)
        action_indices = [self.actions.index(action) for action in predictions]
        plt.plot(action_indices, marker='o')
        plt.yticks(range(len(self.actions)), self.actions)
        plt.title('Predicted Actions Over Time')
        plt.xlabel('Time Window')
        plt.ylabel('Predicted Action')
        
        # Plot commitment levels
        plt.subplot(2, 1, 2)
        plt.plot(commitments, marker='o')
        plt.title('Commitment Levels Over Time')
        plt.xlabel('Time Window')
        plt.ylabel('Commitment Level')
        
        plt.tight_layout()
        plt.show()

    def build_tunable_model(self, hp):
        input_shape = (250, 16)
        input_cnn = Input(shape=input_shape)
        x_cnn = Conv1D(hp.Int('conv_1_filter', min_value=32, max_value=128, step=32),
                       hp.Int('conv_1_kernel', min_value=3, max_value=9, step=3),
                       activation='relu')(input_cnn)
        x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
        x_cnn = Conv1D(hp.Int('conv_2_filter', min_value=64, max_value=256, step=64),
                       hp.Int('conv_2_kernel', min_value=3, max_value=9, step=3),
                       activation='relu')(x_cnn)
        x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
        x_cnn = LSTM(hp.Int('lstm_units', min_value=32, max_value=128, step=32), return_sequences=True)(x_cnn)
        x_cnn = LSTM(hp.Int('lstm_units_2', min_value=16, max_value=64, step=16))(x_cnn)
        
        input_transformer = Input(shape=input_shape)
        x_transformer = input_transformer
        for _ in range(hp.Int('num_transformer_blocks', 1, 3)):
            x_transformer = MultiHeadAttention(num_heads=hp.Int('num_heads', 2, 8),
                                               key_dim=hp.Int('key_dim', 16, 64, step=16))(x_transformer, x_transformer)
            x_transformer = LayerNormalization(epsilon=1e-6)(x_transformer)
            x_transformer = Conv1D(filters=hp.Int('conv_transformer_filters', 16, 64, step=16),
                                   kernel_size=1, activation="relu")(x_transformer)
            x_transformer = LayerNormalization(epsilon=1e-6)(x_transformer)
        x_transformer = GlobalAveragePooling1D()(x_transformer)
        
        combined = Concatenate()([x_cnn, x_transformer])
        combined = Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu')(combined)
        combined = Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1))(combined)
        output = Dense(len(self.actions), activation='softmax')(combined)
        
        model = Model(inputs=[input_cnn, input_transformer], outputs=output)
        model.compile(
            optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        
        return model

    def optimize_nn_model(self, X, y):
        tuner = kt.Hyperband(self.build_tunable_model,
                             objective='val_accuracy',
                             max_epochs=50,
                             factor=3,
                             directory='my_dir',
                             project_name='EEG_model_tuning')

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search([X, X], y, epochs=50, validation_split=0.2, callbacks=[stop_early])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.models['optimized_nn'] = tuner.hypermodel.build(best_hps)
        return self.models['optimized_nn']
    
    def check_data_consistency(self):
        self.log_and_emit("Data Consistency Summary:")
        shapes = {}
        sample_counts = {}
        
        if not self.actions:
            self.log_and_emit("No actions available.")
            return

        for data_type in self.data.keys():
            self.log_and_emit(f"\nChecking {data_type}:")
            shapes[data_type] = {}
            sample_counts[data_type] = {}
            
            for action in self.actions:
                if action not in self.data[data_type]:
                    self.log_and_emit(f"  Warning: No data found for action '{action}' in {data_type}")
                    continue
                shapes[data_type][action] = [sample.shape for sample in self.data[data_type][action]]
                sample_counts[data_type][action] = len(shapes[data_type][action])

            if not shapes[data_type]:
                self.log_and_emit(f"  No valid data found for any action in {data_type}")
                continue

            # Summarize shape information
            for action, action_shapes in shapes[data_type].items():
                unique_shapes = set(action_shapes)
                if len(unique_shapes) == 1:
                    self.log_and_emit(f"  {action}: All {sample_counts[data_type][action]} samples have shape {unique_shapes.pop()}")
                else:
                    self.log_and_emit(f"  {action}: {sample_counts[data_type][action]} samples with {len(unique_shapes)} different shapes")
                    self.log_and_emit(f"    Most common shape: {max(set(action_shapes), key=action_shapes.count)}")
                    self.log_and_emit(f"    Unusual shapes:")
                    for shape in unique_shapes:
                        if action_shapes.count(shape) < 5:  # Arbitrary threshold for "unusual"
                            self.log_and_emit(f"      {shape}: {action_shapes.count(shape)} occurrences")

            # Check if all actions have the same number of samples
            if len(set(sample_counts[data_type].values())) > 1:
                self.log_and_emit(f"\n  Inconsistent number of samples across actions in {data_type}:")
                for action, count in sample_counts[data_type].items():
                    self.log_and_emit(f"    {action}: {count} samples")
            elif sample_counts[data_type]:
                self.log_and_emit(f"\n  All actions in {data_type} have {next(iter(sample_counts[data_type].values()))} samples")
            else:
                self.log_and_emit(f"\n  No sample count information available for {data_type}")

            # Overall statistics for this data type
            all_shapes = [shape for shapes_list in shapes[data_type].values() for shape in shapes_list]
            total_samples = sum(sample_counts[data_type].values())
            
            if total_samples > 0:
                standard_shape = (250, 16, 60)
                standard_count = all_shapes.count(standard_shape)

                self.log_and_emit(f"\n  Total samples in {data_type}: {total_samples}")
                self.log_and_emit(f"  Samples with standard shape {standard_shape}: {standard_count} ({standard_count/total_samples*100:.2f}%)")
                self.log_and_emit(f"  Samples with non-standard shapes: {total_samples - standard_count} ({(total_samples - standard_count)/total_samples*100:.2f}%)")
            else:
                self.log_and_emit(f"\n  No samples found in {data_type}")

    def clean_and_balance_data(self):
        self.log_and_emit("Cleaning and balancing data...")
        if not self.actions or not self.data['raw_data']:
            self.log_and_emit("No raw data available to clean and balance.")
            return

        cleaned_data = {}
        sample_counts = {}

        # Step 1: Remove non-standard shapes
        for action in self.actions:
            if action not in self.data['raw_data']:
                self.log_and_emit(f"No raw data found for action: {action}")
                continue
            cleaned_data[action] = [sample for sample in self.data['raw_data'][action] if sample.shape == (250, 16, 60)]
            sample_counts[action] = len(cleaned_data[action])
            self.log_and_emit(f"{action}: {sample_counts[action]} samples after cleaning")

        if not cleaned_data:
            self.log_and_emit("No valid data remaining after cleaning.")
            return

        # Step 2: Balance the dataset
        min_samples = min(sample_counts.values())
        for action in self.actions:
            if action not in cleaned_data:
                continue
            if len(cleaned_data[action]) > min_samples:
                # Randomly select min_samples
                indices = np.random.choice(len(cleaned_data[action]), min_samples, replace=False)
                cleaned_data[action] = [cleaned_data[action][i] for i in indices]
            self.log_and_emit(f"{action}: {len(cleaned_data[action])} samples after balancing")

        # Update the framework's cleaned data
        self.data['cleaned_data'] = cleaned_data

        # Verify the results
        shapes = {action: [sample.shape for sample in cleaned_data[action]] for action in cleaned_data}
        self.log_and_emit("\nFinal data summary:")
        for action, action_shapes in shapes.items():
            shape_counts = Counter(action_shapes)
            self.log_and_emit(f"{action}:")
            for shape, count in shape_counts.items():
                self.log_and_emit(f"  Shape {shape}: {count} samples")

        self.log_and_emit("Data cleaning and balancing completed.")

    def run_analysis(self):
        self.log_and_emit("Loading data...")
        self.load_data()

        self.log_and_emit("Preprocessing data...")
        self.preprocess_all_data()

        self.log_and_emit("Applying ICA...")
        self.apply_ica()

        self.log_and_emit("Clustering ICA components...")
        self.cluster_ica_components()

        self.log_and_emit("Preparing data for model training...")
        train_X, test_X, train_y, test_y = self.prepare_data_for_training()

        self.log_and_emit("Training and optimizing Combined CNN-LSTM-Transformer model...")
        self.train_combined_model(train_X, train_y, test_X, test_y)

        self.log_and_emit("Training and optimizing Neural Network model...")
        self.optimize_nn_model(train_X, train_y)

        self.log_and_emit("Training and optimizing Random Forest model...")
        self.train_rf_model(train_X.reshape(train_X.shape[0], -1), np.argmax(train_y, axis=1))

        self.log_and_emit("Performing cross-validation...")
        self.cross_validate_models(train_X, train_y)

        self.log_and_emit("Evaluating Combined model...")
        combined_results = self.evaluate_model('combined', test_X, test_y)

        self.log_and_emit("Evaluating Optimized NN model...")
        optimized_nn_results = self.evaluate_model('optimized_nn', test_X, test_y)

        self.log_and_emit("Evaluating Random Forest model...")
        rf_results = self.evaluate_model('rf', test_X.reshape(test_X.shape[0], -1), test_y)

        self.log_and_emit("Analysis complete.")

    def get_results(self):
        return {
            'actions': self.actions,
            'model_performances': {
                'combined': self.evaluate_model('combined', self.preprocessed_data, self.actions),
                'optimized_nn': self.evaluate_model('optimized_nn', self.preprocessed_data, self.actions),
                'rf': self.evaluate_model('rf', self.preprocessed_data, self.actions)
            }
        }