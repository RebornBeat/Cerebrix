from abc import abstractmethod
from .base import BaseFeatureExtractor
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (InputLayer, Dense, LSTM, TimeDistributed,
                                   Reshape, RepeatVector)
import numpy as np
from minisom import MiniSom
import pickle
from typing import List, Optional
import keras.backend as K
from sklearn.cluster import KMeans

class BaseUnsupervisedExtractor(BaseFeatureExtractor):
    def __init__(self, name: str, learning_rate: float = 0.01):
        super().__init__(name)
        self.learning_rate = learning_rate
        self.training_state = {}
        self.name = name
        self.initial_learning_rate = learning_rate
        self.training_history = []
        self.current_performance = 0
        self.previous_improvements = []

    @abstractmethod
    def update_weights_from_performance(self,
                                      performance: float,
                                      features: np.ndarray,
                                      stack_size: Optional[int] = None):
        """Update model weights based on decision layer performance"""
        pass

    def static_train(self, windows: np.ndarray, epochs: int = 100):
        """Pure unsupervised training without integration"""
        raise NotImplementedError

    def integrated_train_step_single_phase(self, windows: np.ndarray, performance: float = None):
        """Original integrated training with larger updates"""
        raise NotImplementedError

    def integrated_train_step_two_phase(self, windows: np.ndarray, performance: float = None):
        """Two-phase approach with initial training + fine-tuning"""
        raise NotImplementedError

    def integrated_train_step_adaptive(self, windows: np.ndarray, performance: float = None):
        """Adaptive updates based on performance patterns"""
        raise NotImplementedError


class AutoencoderExtractor(BaseUnsupervisedExtractor):
    def __init__(self,
                 window_size: int,
                 latent_dim: int = 32,
                 stack_size: int = 1):
        super().__init__("autoencoder")
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.stack_size = stack_size
        self.autoencoder, self.encoder = self._build_model()

    def _build_model(self):
        """Build autoencoder model"""
        # Input shape: (window_size, 16 channels, 60 frequencies * stack_size)
        input_shape = (self.window_size, 16, 60 * self.stack_size)
        inputs = Input(shape=input_shape)

        # Encoder
        # Process each timestep independently first
        x = TimeDistributed(Dense(32, activation='relu'))(inputs)

        # Sequential processing with LSTM
        x = LSTM(64, return_sequences=True)(x)
        encoded = LSTM(self.latent_dim, return_sequences=False)(x)

        # Decoder
        # Gradually rebuild the signal
        x = Dense(64, activation='relu')(encoded)
        x = Dense(128, activation='relu')(x)

        # Reshape back to original dimensions
        decoded = Dense(self.window_size * 16 * 60, activation='linear')(x)
        decoded = Reshape((self.window_size, 16, 60))(decoded)

        # Create models
        autoencoder = Model(inputs, decoded, name='autoencoder')
        encoder = Model(inputs, encoded, name='encoder')

        # Compile autoencoder
        autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']  # Added metric for monitoring
        )

        return autoencoder, encoder

    def static_train(self,
              windows: np.ndarray,
              epochs: int = 10,
              batch_size: int = 32,
              validation_split: float = 0.2):
        """Train the autoencoder"""
        return self.autoencoder.fit(
            windows,
            windows,  # Autoencoder reconstructs its input
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )

    def extract(self, window: np.ndarray) -> np.ndarray:
        """Extract features using encoder"""
        # Ensure correct input shape
        if len(window.shape) == 3:
            window = window.reshape(1, *window.shape)

        # Get encoded representation
        encoded_features = self.encoder.predict(window)
        return encoded_features

    def get_feature_names(self) -> List[str]:
        return [f"ae_feature_{i}" for i in range(self.latent_dim)]

    def reconstruct(self, window: np.ndarray) -> np.ndarray:
        """Reconstruct input from autoencoder for validation"""
        return self.autoencoder.predict(window)

    def update_weights_from_performance(self,
                                      performance: float,
                                      features: np.ndarray,
                                      stack_size: Optional[int] = None):
        """Update autoencoder weights based on decision layer performance"""
        improvement = performance - self.current_performance
        if improvement > 0:
            weight_update = self.learning_rate * improvement
            self._apply_weight_updates(weight_update)
        self.current_performance = performance

    def _apply_weight_updates(self, weight_update: float):
        """Apply weight updates to model layers"""
        for layer in self.autoencoder.layers:
            if hasattr(layer, 'kernel'):
                layer.kernel.assign_add(layer.kernel * weight_update)
            if hasattr(layer, 'bias'):
                layer.bias.assign_add(layer.bias * weight_update)

    def integrated_train_step_single_phase(self, windows: np.ndarray, performance: float = None):
        """Original integrated approach with larger updates"""
        # Regular autoencoder training
        self.autoencoder.train_on_batch(windows, windows)

        if performance is not None:
            improvement = performance - self.current_performance
            if improvement > 0:
                # Larger weight updates
                weight_update = self.learning_rate * improvement
                self._apply_weight_updates(weight_update)
            self.current_performance = performance

        # Return extracted features
        return self.encoder.predict(windows)

    def integrated_train_step_two_phase(self, windows: np.ndarray, performance: float = None):
        """Two-phase approach with initial training + fine-tuning"""
        if not self.is_initialized:
            self.initial_train(windows, epochs=50)
            self.is_initialized = True

        # Regular autoencoder training with smaller updates
        self.autoencoder.train_on_batch(windows, windows)

        if performance is not None:
            improvement = performance - self.current_performance
            if improvement > 0:
                # Smaller updates for fine-tuning
                weight_update = self.learning_rate * improvement * 0.5
                self._apply_weight_updates(weight_update)
            self.current_performance = performance

        # Extract features using encoder
        return self.encoder.predict(windows)

    def integrated_train_step_adaptive(self, windows: np.ndarray, performance: float = None):
        """Adaptive update approach"""
        self.autoencoder.train_on_batch(windows, windows)

        if performance is not None:
            improvement = performance - self.current_performance
            if improvement > 0:
                # Adapt update size based on improvement
                if improvement > 0.1:  # Significant improvement
                    weight_update = self.learning_rate * improvement
                else:  # Small improvement
                    weight_update = self.learning_rate * improvement * 0.5

                # Add momentum for consistent improvements
                if len(self.previous_improvements) >= 3 and \
                   all(imp > 0 for imp in self.previous_improvements[-3:]):
                    weight_update *= 1.2

                self._apply_weight_updates(weight_update)

                # Update history
                self.previous_improvements.append(improvement)
                if len(self.previous_improvements) > 5:
                    self.previous_improvements.pop(0)

                # Adjust learning rate based on stability
                if len(self.previous_improvements) >= 3:
                    stability = np.std(self.previous_improvements[-3:])
                    if stability > 0.2:  # High variance
                        self.learning_rate *= 0.9

            self.current_performance = performance

        return self.encoder.predict(windows)

    def initial_train(self, windows: np.ndarray, epochs: int = 50):
        """Initial training for two-phase approach"""
        return self.autoencoder.fit(
            windows,
            windows,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )

    def save_model(self, path: str):
        self.autoencoder.save(path)

    def load_model(self, path: str):
        self.autoencoder = tf.keras.models.load_model(path)
        # Recreate encoder from loaded autoencoder
        self.encoder = Model(
            inputs=self.autoencoder.input,
            outputs=self.autoencoder.get_layer('encoder').output
        )

class RecurrentAutoencoder(BaseUnsupervisedExtractor):
    """Pure LSTM-based autoencoder"""
    def __init__(self, window_size: int, latent_dim: int = 32, stack_size: int = 1):
        super().__init__("recurrent_autoencoder")
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.stack_size = stack_size
        self.autoencoder, self.encoder = self._build_model()
        self.is_initialized = False

    def _build_model(self):
        input_shape = (self.window_size, 16, 60 * self.stack_size)
        inputs = Input(shape=input_shape)

        # Encoder
        x = LSTM(64, return_sequences=True)(inputs)
        encoded = LSTM(self.latent_dim, return_sequences=False)(x)

        # Decoder
        x = RepeatVector(self.window_size)(encoded)
        x = LSTM(32, return_sequences=True)(x)
        decoded = TimeDistributed(Dense(16 * 60))(x)
        decoded = Reshape((self.window_size, 16, 60))(decoded)

        # Create models
        autoencoder = Model(inputs, decoded, name='recurrent_autoencoder')
        encoder = Model(inputs, encoded, name='recurrent_encoder')

        # Compile
        autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        return autoencoder, encoder

    def reconstruct(self, window: np.ndarray) -> np.ndarray:
        """Reconstruct input from autoencoder for validation"""
        return self.autoencoder.predict(window)

    def update_weights_from_performance(self,
                                      performance: float,
                                      features: np.ndarray,
                                      stack_size: Optional[int] = None):
        """Update reccurent_autoencoder weights based on decision layer performance"""
        improvement = performance - self.current_performance
        if improvement > 0:
            weight_update = self.learning_rate * improvement
            self._apply_weight_updates(weight_update)
        self.current_performance = performance

    def _apply_weight_updates(self, weight_update: float):
        """Apply weight updates to model layers"""
        for layer in self.autoencoder.layers:
            if hasattr(layer, 'kernel'):
                layer.kernel.assign_add(layer.kernel * weight_update)
            if hasattr(layer, 'bias'):
                layer.bias.assign_add(layer.bias * weight_update)

    def static_train(self, windows: np.ndarray, epochs: int = 100):
        """Traditional recurrent autoencoder training"""
        return self.autoencoder.fit(
            windows,
            windows,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )

    def integrated_train_step_single_phase(self, windows: np.ndarray, performance: float = None):
        """Original integrated approach"""
        self.autoencoder.train_on_batch(windows, windows)

        if performance is not None:
            improvement = performance - self.current_performance
            if improvement > 0:
                # Larger updates
                weight_update = self.learning_rate * improvement
                self._apply_lstm_updates(weight_update)
            self.current_performance = performance

        return self.encoder.predict(windows)

    def integrated_train_step_two_phase(self, windows: np.ndarray, performance: float = None):
        """Two-phase approach"""
        if not self.is_initialized:
            self.initial_train(windows, epochs=50)
            self.is_initialized = True

        self.autoencoder.train_on_batch(windows, windows)

        if performance is not None:
            improvement = performance - self.current_performance
            if improvement > 0:
                # Smaller updates for fine-tuning
                weight_update = self.learning_rate * improvement * 0.5
                self._apply_lstm_updates(weight_update)
            self.current_performance = performance

        return self.encoder.predict(windows)

    def integrated_train_step_adaptive(self, windows: np.ndarray, performance: float = None):
        """Adaptive update approach"""
        self.autoencoder.train_on_batch(windows, windows)

        if performance is not None:
            improvement = performance - self.current_performance
            if improvement > 0:

                # Adaptive update size
                if improvement > 0.1:  # Significant improvement
                    weight_update = self.learning_rate * improvement
                else:  # Small improvement
                    weight_update = self.learning_rate * improvement * 0.5

                # Add momentum for consistent improvements
                if len(self.previous_improvements) >= 3 and \
                   all(imp > 0 for imp in self.previous_improvements[-3:]):
                    weight_update *= 1.2

                self._apply_lstm_updates(weight_update)

                # Update history and adjust learning rate
                self.previous_improvements.append(improvement)
                if len(self.previous_improvements) > 5:
                    self.previous_improvements.pop(0)

                if len(self.previous_improvements) >= 3:
                    stability = np.std(self.previous_improvements[-3:])
                    if stability > 0.2:  # High variance
                        self.learning_rate *= 0.9

            self.current_performance = performance

        return self.encoder.predict(windows)

    def _apply_lstm_updates(self, weight_update):
        """Apply updates to LSTM layers"""
        for layer in self.autoencoder.layers:
            if isinstance(layer, tf.keras.layers.LSTM):
                weights = layer.get_weights()
                for i in range(len(weights)):
                    weights[i] += weight_update * weights[i]
                layer.set_weights(weights)

    def initial_train(self, windows: np.ndarray, epochs: int = 50):
        """Initial training for two-phase approach"""
        return self.autoencoder.fit(
            windows,
            windows,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )

    def save_model(self, path: str):
        self.autoencoder.save(path)

    def load_model(self, path: str):
        self.autoencoder = tf.keras.models.load_model(path)
        self.encoder = Model(
            inputs=self.autoencoder.input,
            outputs=self.autoencoder.get_layer('recurrent_encoder').output
        )


class DECExtractor(BaseUnsupervisedExtractor):
    def __init__(self,
                 window_size: int,
                 n_clusters: int = 5,
                 stack_size: int = 1,
                 latent_dim: int = 32):
        super().__init__("dec")
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.stack_size = stack_size
        self.latent_dim = latent_dim
        self.model = self._build_model()

    def _build_model(self):
        """Build DEC model with proper architecture"""
        # Input shape: (window_size, 16 channels, 60 frequencies * stack_size)
        input_shape = (self.window_size, 16, 60 * self.stack_size)
        inputs = Input(shape=input_shape)

        # Feature extraction layers
        x = TimeDistributed(Dense(64, activation='relu'))(inputs)
        x = LSTM(32, return_sequences=False)(x)

        # Clustering layer
        clustering_layer = ClusteringLayer(
            n_clusters=self.n_clusters,
            name='clustering'
        )(x)

        # Build and compile model
        model = Model(inputs=inputs, outputs=clustering_layer)
        model.compile(
            optimizer='adam',
            loss='kld'  # Kullback-Leibler divergence loss
        )

        return model

    def get_feature_names(self) -> List[str]:
        return [f"cluster_prob_{i}" for i in range(self.n_clusters)]

    class ClusteringLayer(tf.keras.layers.Layer):
        def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
            super().__init__(**kwargs)
            self.n_clusters = n_clusters
            self.alpha = alpha
            self.initial_weights = weights

        def build(self, input_shape):
            self.clusters = self.add_weight(
                shape=(self.n_clusters, input_shape[-1]),
                initializer='glorot_uniform',
                name='clusters'
            )
            if self.initial_weights is not None:
                self.set_weights(self.initial_weights)

        def call(self, inputs):
            """ Student t-distribution kernel """
            q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) -
                                            self.clusters), axis=2) / self.alpha))
            q = q ** ((self.alpha + 1.0) / 2.0)
            q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
            return q

    def extract(self, window: np.ndarray) -> np.ndarray:
        """Extract cluster membership features"""
        if len(window.shape) == 3:
            window = window.reshape(1, *window.shape)
        return self.model.predict(window)

    def update_weights_from_performance(self,
                                      performance: float,
                                      features: np.ndarray,
                                      stack_size: Optional[int] = None):
        """Update DEC weights based on decision layer performance"""
        improvement = performance - self.current_performance
        if improvement > 0:
            cluster_centers = self.model.get_layer('clustering').get_weights()[0]
            features_pred = self.model.predict(features)
            weight_update = self.learning_rate * improvement
            new_centers = cluster_centers + weight_update * (features_pred.T @ (features - cluster_centers))
            self.model.get_layer('clustering').set_weights([new_centers])
        self.current_performance = performance

    def static_train(self,
              windows: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              update_interval: int = 140):
        """Train DEC model"""
        # Initial cluster centers using k-means
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        features = self.model.layers[-2].output  # Get features before clustering layer
        y_pred = kmeans.fit_predict(features)

        # Initialize cluster centers
        self.model.get_layer('clustering').set_weights([kmeans.cluster_centers_])

        # Compute target distribution
        q = self.model.predict(windows)
        p = self._target_distribution(q)

        # Train model
        return self.model.fit(
            windows,
            p,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )

    def _target_distribution(self, q):
        """Compute target distribution P from Q"""
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def get_cluster_assignments(self, window: np.ndarray):
        """Get hard cluster assignments"""
        q = self.extract(window)
        return np.argmax(q, axis=1)

    def integrated_train_step_single_phase(self, windows: np.ndarray, performance: float = None):
        """Original integrated approach"""
        # Get current cluster assignments
        q = self.model.predict(windows)
        p = self._target_distribution(q)

        # Update clustering
        self.model.train_on_batch(windows, p)

        if performance is not None:
            improvement = performance - self.current_performance
            if improvement > 0:
                # Larger updates
                cluster_centers = self.model.get_layer('clustering').get_weights()[0]
                weight_update = self.learning_rate * improvement
                features_pred = self.model.predict(windows)
                new_centers = cluster_centers + weight_update * (features_pred.T @ (windows - cluster_centers))
                self.model.get_layer('clustering').set_weights([new_centers])
            self.current_performance = performance

        return self.model.predict(windows)

    def integrated_train_step_two_phase(self, windows: np.ndarray, performance: float = None):
        """Two-phase approach"""
        if not self.is_initialized:
            self.initial_train(windows, epochs=50)
            self.is_initialized = True

        # Fine-tuning phase
        q = self.model.predict(windows)
        p = self._target_distribution(q)
        self.model.train_on_batch(windows, p)

        if performance is not None:
            improvement = performance - self.current_performance
            if improvement > 0:
                # Smaller updates for fine-tuning
                cluster_centers = self.model.get_layer('clustering').get_weights()[0]
                weight_update = self.learning_rate * improvement * 0.5
                features_pred = self.model.predict(windows)
                new_centers = cluster_centers + weight_update * (features_pred.T @ (windows - cluster_centers))
                self.model.get_layer('clustering').set_weights([new_centers])
            self.current_performance = performance

        return self.model.predict(windows)

    def integrated_train_step_adaptive(self, windows: np.ndarray, performance: float = None):
        """Adaptive update approach"""
        q = self.model.predict(windows)
        p = self._target_distribution(q)
        self.model.train_on_batch(windows, p)

        if performance is not None:
            improvement = performance - self.current_performance
            if improvement > 0:
                cluster_centers = self.model.get_layer('clustering').get_weights()[0]
                features_pred = self.model.predict(windows)

                # Adaptive update size
                if improvement > 0.1:  # Significant improvement
                    weight_update = self.learning_rate * improvement
                else:  # Small improvement
                    weight_update = self.learning_rate * improvement * 0.5

                # Add momentum for consistent improvements
                if len(self.previous_improvements) >= 3 and \
                   all(imp > 0 for imp in self.previous_improvements[-3:]):
                    weight_update *= 1.2

                new_centers = cluster_centers + weight_update * (features_pred.T @ (windows - cluster_centers))
                self.model.get_layer('clustering').set_weights([new_centers])

                # Update history and adjust learning rate
                self.previous_improvements.append(improvement)
                if len(self.previous_improvements) > 5:
                    self.previous_improvements.pop(0)

                if len(self.previous_improvements) >= 3:
                    stability = np.std(self.previous_improvements[-3:])
                    if stability > 0.2:  # High variance
                        self.learning_rate *= 0.9

            self.current_performance = performance

        return self.model.predict(windows)

    def initial_train(self, windows: np.ndarray, epochs: int = 50):
        """Initial training for two-phase approach"""
        # Initial cluster centers using k-means
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        features = self.model.layers[-2].output  # Get features before clustering layer
        y_pred = kmeans.fit_predict(features)

        # Initialize cluster centers
        self.model.get_layer('clustering').set_weights([kmeans.cluster_centers_])

        # Compute initial target distribution
        q = self.model.predict(windows)
        p = self._target_distribution(q)

        # Train model
        return self.model.fit(
            windows,
            p,
            epochs=epochs,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )

    def save_model(self, path: str):
        self.model.save(path)

    def load_model(self, path: str):
        self.model = tf.keras.models.load_model(
            path,
            custom_objects={'ClusteringLayer': self.ClusteringLayer}
        )

class SOMExtractor(BaseUnsupervisedExtractor):
    def __init__(self,
                 window_size: int,
                 stack_size: int = 1,
                 map_size: tuple = (10, 10),
                 sigma: float = 1.0,
                 learning_rate: float = 0.5,
                 neighborhood_function: str = 'gaussian',
                 topology: str = 'hexagonal'):  # 'hexagonal' or 'rectangular'
        super().__init__("som")
        self.window_size = window_size
        self.stack_size = stack_size
        self.map_size = map_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.neighborhood_function = neighborhood_function
        self.topology = topology
        self.input_len = self.window_size * 16 * 60 * self.stack_size

        self.som = self._initialize_som()
        self.trained = False

    def _initialize_som(self):
        """Initialize SOM with EEG-specific configuration"""
        return MiniSom(
            x=self.map_size[0],
            y=self.map_size[1],
            input_len=self.input_len,
            sigma=self.sigma,
            learning_rate=self.learning_rate,
            neighborhood_function=self.neighborhood_function,
            topology=self.topology
        )

    def _preprocess_window(self, window: np.ndarray) -> np.ndarray:
        """Preprocess window for SOM"""
        # Normalize
        normalized = (window - window.mean()) / (window.std() + 1e-8)

        # Reshape for SOM
        if len(normalized.shape) == 4:  # Batch of windows
            return normalized.reshape(normalized.shape[0], -1)
        return normalized.reshape(1, -1)

    def update_weights_from_performance(self,
                                      performance: float,
                                      features: np.ndarray,
                                      stack_size: Optional[int] = None):
        """Update SOM weights based on decision layer performance"""
        improvement = performance - self.current_performance
        if improvement > 0:
            # Adjust learning rate based on performance
            adjusted_lr = self.learning_rate * (1 + improvement)
            self.som.update_learning_rate(adjusted_lr)
            # Update weights with adjusted learning rate
            for feature in features:
                winner = self.som.winner(feature)
                self.som.update(feature, winner, adjusted_lr)
        self.current_performance = performance

    def static_train(self,
             windows: np.ndarray,
             epochs: int = 100,
             batch_size: int = 32,
             shuffle: bool = True):
        """Train SOM on EEG windows"""
        processed_data = self._preprocess_window(windows)

        # Initialize progress tracking
        total_iterations = epochs * len(processed_data) // batch_size
        current_iteration = 0

        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(processed_data)

            for idx in range(0, len(processed_data), batch_size):
                batch = processed_data[idx:idx + batch_size]

                # Update learning rate and neighborhood size
                current_lr = self.learning_rate * (1 - current_iteration/total_iterations)
                current_sigma = self.sigma * (1 - current_iteration/total_iterations)

                # Train on batch
                for sample in batch:
                    self.som.update(sample,
                                  self.som.winner(sample),
                                  current_lr,
                                  current_sigma)

                current_iteration += 1

        self.trained = True

    def integrated_train_step_single_phase(self, windows: np.ndarray, performance: float = None):
        """Original integrated approach"""
        processed_data = self._preprocess_window(windows)
        quantization_error = 0

        # Regular SOM update
        for sample in processed_data:
            winner = self.som.winner(sample)
            self.som.update(sample, winner, self.learning_rate)
            quantization_error += self.som.quantization_error([sample])

        if performance is not None:
            improvement = performance - self.current_performance
            if improvement > 0:
                # Larger updates
                weight_update = self.learning_rate * improvement
                self.som.update_learning_rate(self.learning_rate + weight_update)
            self.current_performance = performance

        return self.extract(windows)

    def integrated_train_step_two_phase(self, windows: np.ndarray, performance: float = None):
        """Two-phase approach"""
        if not self.is_initialized:
            self.initial_train(windows, epochs=50)
            self.is_initialized = True

        processed_data = self._preprocess_window(windows)
        quantization_error = 0

        for sample in processed_data:
            winner = self.som.winner(sample)
            # Smaller updates for fine-tuning
            current_lr = self.learning_rate * 0.5
            self.som.update(sample, winner, current_lr)
            quantization_error += self.som.quantization_error([sample])

        if performance is not None:
            improvement = performance - self.current_performance
            if improvement > 0:
                # Smaller updates for fine-tuning
                weight_update = self.learning_rate * improvement * 0.5
                self.som.update_learning_rate(self.learning_rate + weight_update)
            self.current_performance = performance

        return self.extract(windows)

    def integrated_train_step_adaptive(self, windows: np.ndarray, performance: float = None):
        """Adaptive update approach"""
        processed_data = self._preprocess_window(windows)
        quantization_error = 0

        for sample in processed_data:
            winner = self.som.winner(sample)
            self.som.update(sample, winner, self.learning_rate)
            quantization_error += self.som.quantization_error([sample])

        if performance is not None:
            improvement = performance - self.current_performance
            if improvement > 0:
                # Adaptive update size
                if improvement > 0.1:  # Significant improvement
                    weight_update = self.learning_rate * improvement
                else:  # Small improvement
                    weight_update = self.learning_rate * improvement * 0.5

                # Add momentum for consistent improvements
                if len(self.previous_improvements) >= 3 and \
                   all(imp > 0 for imp in self.previous_improvements[-3:]):
                    weight_update *= 1.2

                self.som.update_learning_rate(self.learning_rate + weight_update)

                # Update history and adjust learning rate
                self.previous_improvements.append(improvement)
                if len(self.previous_improvements) > 5:
                    self.previous_improvements.pop(0)

                if len(self.previous_improvements) >= 3:
                    stability = np.std(self.previous_improvements[-3:])
                    if stability > 0.2:  # High variance
                        self.learning_rate *= 0.9

            self.current_performance = performance

        return self.extract(windows)

    def _preprocess_window(self, window: np.ndarray) -> np.ndarray:
        """Preprocess window for SOM"""
        normalized = (window - window.mean()) / (window.std() + 1e-8)
        if len(normalized.shape) == 4:  # Batch of windows
            return normalized.reshape(normalized.shape[0], -1)
        return normalized.reshape(1, -1)

    def extract(self, window: np.ndarray) -> np.ndarray:
        """Extract SOM features from window"""
        if not self.trained:
            raise ValueError("SOM must be trained before feature extraction")

        processed = self._preprocess_window(window)
        features = []

        for sample in processed:
            # Get activation map
            activation_map = self.som.activate(sample)

            # Get winner neuron
            winner = self.som.winner(sample)

            # Get distance map
            distance_map = self.som.distance_map()

            # Compute topographic error
            topo_error = self._compute_topographic_error(sample)

            # Combine features
            sample_features = np.concatenate([
                activation_map.flatten(),
                np.array(winner),
                [np.mean(distance_map), np.std(distance_map)],
                [topo_error]
            ])

            features.append(sample_features)

        return np.array(features)

    def _compute_topographic_error(self, sample: np.ndarray) -> float:
        """Compute topographic error for quality assessment"""
        winner = self.som.winner(sample)
        second_winner = self._find_second_best(sample)

        if self.topology == 'hexagonal':
            return self._hexagonal_distance(winner, second_winner)
        return self._euclidean_distance(winner, second_winner)

    def _find_second_best(self, sample: np.ndarray):
        """Find second best matching unit"""
        activation_map = self.som.activate(sample)
        winner_value = activation_map[self.som.winner(sample)]
        activation_map[self.som.winner(sample)] = np.inf
        return np.unravel_index(np.argmin(activation_map), activation_map.shape)

    def get_feature_names(self) -> List[str]:
        """Get names of SOM features"""
        names = []

        # Activation map features
        total_nodes = self.map_size[0] * self.map_size[1]
        names.extend([f"som_activation_{i}" for i in range(total_nodes)])

        # Winner coordinates
        names.extend(["winner_x", "winner_y"])

        # Distance map statistics
        names.extend(["distance_map_mean", "distance_map_std"])

        # Topographic error
        names.append("topographic_error")

        return names

    def visualize_map(self, save_path: Optional[str] = None):
        """Visualize SOM mapping"""
        plt.figure(figsize=(10, 10))

        # Plot distance map
        plt.subplot(221)
        plt.title('Distance Map')
        plt.imshow(self.som.distance_map(), cmap='bone_r')
        plt.colorbar()

        # Plot activation patterns
        plt.subplot(222)
        plt.title('Activation Patterns')
        plt.imshow(self.som.activation_response(), cmap='viridis')
        plt.colorbar()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def get_cluster_labels(self, windows: np.ndarray) -> np.ndarray:
        """Get cluster labels for windows"""
        processed = self._preprocess_window(windows)
        return np.array([self.som.winner(sample) for sample in processed])

    def initial_train(self, windows: np.ndarray, epochs: int = 50):
        """Initial training for two-phase approach"""
        processed_data = self._preprocess_window(windows)

        # Initialize progress tracking
        total_iterations = epochs * len(processed_data)
        current_iteration = 0

        for epoch in range(epochs):
            # Shuffle data
            np.random.shuffle(processed_data)

            for sample in processed_data:
                # Calculate learning rate and sigma decay
                progress = current_iteration / total_iterations
                current_lr = self.learning_rate * (1 - progress)
                current_sigma = self.sigma * (1 - progress)

                # Update SOM
                winner = self.som.winner(sample)
                self.som.update(sample, winner, current_lr, current_sigma)
                current_iteration += 1

        self.trained = True  # Mark as trained

    def save_model(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.som, f)

    def load_model(self, path: str):
        with open(path, 'rb') as f:
            self.som = pickle.load(f)
