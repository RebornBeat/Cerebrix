import os
import tensorflow as tf
import numpy as np
from collections import defaultdict
from typing import Dict
import logging
import mlflow
from util.evaluation.model_evaluator import ModelEvaluator
from util.evaluation.feature_evaluator import FeatureEvaluator
from util.feature_extractors.unsupervised_extractors import (
    AutoencoderExtractor,
    RecurrentAutoencoder,
    DECExtractor,
    SOMExtractor
)
from util.feature_extractors.temporal_spectral_extractor import HandcraftedFeatureExtractor
from util.feature_extractors.cnn_extractor import CNNFeatureExtractor
from util.training.hyperparameter_tuner import ModelTuner

class IntegratedTrainer:
    def __init__(self,
                 model_class,
                 use_features: bool = False,
                 use_hyperparameter_tuning: bool = False,
                 use_unsupervised_updates: bool = False,
                 stack_specific: bool = False,
                 max_stack_size: int = 3,
                 approach: str = 'adaptive',
                 window_processor=None,
                 **kwargs):
        self.model_class = model_class
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.use_unsupervised_updates = use_unsupervised_updates
        self.stack_specific = stack_specific
        self.max_stack_size = max_stack_size
        self.window_size = 5
        self.use_features = use_features
        self.approach = approach
        self.window_processor = window_processor

        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Track if static models need training
        self.static_models_loaded = False

        if use_features:
            # Initialize feature extractors
            self.initialize_extractors()

        # Initialize evaluators
        self.model_evaluator = ModelEvaluator()
        self.feature_evaluator = FeatureEvaluator()
        self.current_performance = 0

        # Training metrics
        self.training_history = defaultdict(list)

        if use_hyperparameter_tuning:
            self.tuner = ModelTuner(model_class)

        # Initialize models dictionary
        self.models = {}
        self.num_actions = 0

    def log_and_emit(self, message: str, level: int = logging.INFO) -> None:
        """Log message and emit to mlflow if in active run"""
        self.logger.log(level, message)
        if mlflow.active_run():
            mlflow.log_param("message", message)

    def log_epoch_metrics(self, epoch: int, metrics: Dict):
        """Log metrics for epoch"""
        for metric_name, value in metrics.items():
            # Log to mlflow
            mlflow.log_metric(f"epoch_{epoch}_{metric_name}", value)
            # Add to history
            self.training_history[metric_name].append(value)
            # Log to console
            self.log_and_emit(f"Epoch {epoch} - {metric_name}: {value:.4f}")

    def build_models(self):
        """Build models based on configuration"""
        if self.stack_specific:
            for stack_size in range(1, self.max_stack_size + 1):
                self.models[stack_size] = self.model_class(
                    num_classes=self.num_actions,
                    feature_extractor=self.use_features,
                    stack_size=stack_size
                ).build()
        else:
            self.models[1] = self.model_class(
                num_classes=self.num_actions,
                feature_extractor=self.use_features
            ).build()

    def save_models(self, models_dir: str, identifier: str):
        """Save all models centrally"""
        # Save architecture models
        for model_type, model in self.models.items():
            if isinstance(model, tf.keras.Model):
                model_path = os.path.join(models_dir, f'{model_type}_{identifier}.h5')
                model.save(model_path)
                mlflow.keras.log_model(model, f"{model_type}_{identifier}")

        # Save unsupervised models if using features and not static approach
        if self.use_features and self.approach != 'static':
            unsupervised_dir = os.path.join(models_dir, 'unsupervised', self.approach)
            os.makedirs(unsupervised_dir, exist_ok=True)

            for name, extractor in self.extractors['unsupervised'].items():
                model_path = os.path.join(unsupervised_dir, f'{name}_{identifier}.h5')
                extractor.save_model(model_path)
                mlflow.log_artifact(model_path, f"unsupervised_{name}")

    def initialize_extractors(self):
        """Initialize all feature extractors"""
        # Always initialize basic extractors
        self.extractors = {
            'handcrafted': HandcraftedFeatureExtractor(),
            'cnn': CNNFeatureExtractor(),
            'unsupervised': {}
        }

        # Initialize unsupervised extractors based on approach
        if self.approach == 'static':
            self._initialize_static_extractors()
        else:
            self._initialize_dynamic_extractors()

    def _initialize_static_extractors(self):
        """Initialize static models and check if they're pre-trained"""
        static_models_dir = 'static_models'
        os.makedirs(static_models_dir, exist_ok=True)

        all_models_loaded = True
        self.extractors['unsupervised'] = {}

        for name, model_class in {
            'autoencoder': AutoencoderExtractor,
            'recurrent': RecurrentAutoencoder,
            'dec': DECExtractor,
            'som': SOMExtractor
        }.items():
            model_path = os.path.join(static_models_dir, f'{name}_static.h5')

            # Initialize extractor
            extractor = model_class(window_size=self.window_size)

            # Try to load pre-trained model
            if os.path.exists(model_path):
                self.log_and_emit(f"Loading pre-trained static {name} model")
                extractor.load_model(model_path)
            else:
                all_models_loaded = False

            self.extractors['unsupervised'][name] = extractor

        self.static_models_loaded = all_models_loaded

    def _initialize_dynamic_extractors(self):
        """Initialize extractors for integrated training approaches"""
        models_dir = f'model_data/{self.approach}'
        os.makedirs(models_dir, exist_ok=True)

        self.extractors['unsupervised'] = {}
        for name, model_class in {
            'autoencoder': AutoencoderExtractor,
            'recurrent': RecurrentAutoencoder,
            'dec': DECExtractor,
            'som': SOMExtractor
        }.items():
            # Initialize with approach-specific configuration
            self.extractors['unsupervised'][name] = model_class(
                window_size=self.window_size,
                approach=self.approach
            )

    def _train_static_models(self, initial_batch):
        """Train static models if not already trained"""
        self.log_and_emit("Training static feature extractors...")

        windows, _ = initial_batch
        static_models_dir = 'static_models'

        for name, extractor in self.extractors['unsupervised'].items():
            if not hasattr(extractor, 'is_trained') or not extractor.is_trained:
                self.log_and_emit(f"Training static {name} model")
                extractor.static_train(windows)
                model_path = os.path.join(static_models_dir, f'{name}_static.h5')
                extractor.save_model(model_path)
                extractor.is_trained = True

        self.static_models_loaded = True

    def extract_features(self, windows, stack_size=None):
        """Extract features from all extractors"""
        features = {}

        # 1. Handcrafted features
        features['handcrafted'] = self.extractors['handcrafted'].extract(windows)

        # 2. CNN features
        features['cnn'] = self.extractors['cnn'].extract(windows)

        # 3. Unsupervised features - based on training approach
        if self.approach != 'static':
            # Use the appropriate training approach for each unsupervised extractor
            for name, extractor in self.extractors['unsupervised'].items():
                if self.approach == 'single_phase':
                    features[name] = extractor.integrated_train_step_single_phase(
                        windows,
                        self.current_performance
                    )
                elif self.approach == 'two_phase':
                    features[name] = extractor.integrated_train_step_two_phase(
                        windows,
                        self.current_performance
                    )
                elif self.approach == 'adaptive':
                    features[name] = extractor.integrated_train_step_adaptive(
                        windows,
                        self.current_performance
                    )
        else:
            # Use pre-trained static models for feature extraction
            for name, extractor in self.extractors['unsupervised'].items():
                features[name] = extractor.extract(windows)

        return features

    def train(self, train_generator, val_generator, epochs=10, batch_size=32):
        """Main training loop"""
        with mlflow.start_run(run_name=f"training_{self.approach}"):

            # Train static models if needed
            if self.approach == 'static' and not self.static_models_loaded:
                # Get first batch properly
                for batch_windows, batch_labels in train_generator():
                    self._train_static_models((batch_windows, batch_labels))
                    self.num_actions = batch_labels.shape[1]
                    break

                self.build_models()

            # Main training loop
            for epoch in range(epochs):
                self.log_and_emit(f"Epoch {epoch+1}/{epochs}")

                # Training
                for batch_windows, batch_labels in train_generator():
                    if self.stack_specific:
                        self._train_stack(batch_windows, batch_labels)
                    else:
                        self._train_single(batch_windows, batch_labels)

                # Validation using ModelEvaluator
                val_metrics = self.model_evaluator.evaluate_epoch(
                    self.models,
                    val_generator,
                    self.max_stack_size if self.stack_specific else None
                )
                self.log_epoch_metrics(epoch, val_metrics)

        # Save models after training
        self.save_models(os.path.join('model_data', self.approach), f'epoch_{epochs}')

        return self.models

    def _train_stack(self, windows, labels):
        """Train stack-specific models"""

        for stack_size in range(1, self.max_stack_size + 1):
            # Create stacked windows
            stacked_windows = self.window_processor.create_stacked_windows(windows, stack_size)

            # Extract features if using feature extractor
            if self.use_features:
                # Extract features and get metrics
                features = self.extract_features(stacked_windows, stack_size)
                # Combine all features
                combined_features = np.concatenate([feat for feat in features.values()], axis=-1)
                model_input = [stacked_windows, combined_features]
            else:
                model_input = stacked_windows

            # Hyperparameter tuning if enabled
            if self.use_hyperparameter_tuning:
                best_hps = self.tuner.tune_model(
                    model_input,
                    validation_data=None,  # Handle validation data
                    stack_size=stack_size
                )
                self.models[stack_size] = self.model_class(
                    stack_size=stack_size,
                    **best_hps.values
                )

            # Train model
            history = self.models[stack_size].fit(
                model_input,
                labels,
                epochs=1,  # Single epoch per batch
                verbose=0
            )

            # Update current performance for feature extractors
            self.current_performance = history.history['accuracy'][-1]

    def _train_single(self, windows, labels):
        """Train single model"""

        # Extract features if using feature extractor
        if self.use_features:
            features = self.extract_features(windows)
            model_input = [windows, features]
        else:
            model_input = windows

        # Hyperparameter tuning if enabled
        if self.use_hyperparameter_tuning:
            best_hps = self.tuner.tune_model(
                model_input,
                validation_data=None  # Handle validation data
            )
            self.models[1] = self.model_class(**best_hps.values)

        # Train model
        history = self.models[1].fit(
            model_input,
            labels,
            epochs=1,  # Single epoch per batch
            verbose=0
        )

        # Update current performance for feature extractors
        self.current_performance = history.history['accuracy'][-1]

    def evaluate(self, val_generator):
        """Evaluate both models and features"""
        evaluation_results = {
            'model_metrics': {},
            'feature_metrics': {},
            'unsupervised_metrics': {}
        }

        # Model evaluation
        model_metrics = self.model_evaluator.evaluate_epoch(
            self.models,
            val_generator,
            self.max_stack_size if self.stack_specific else None
        )
        evaluation_results['model_metrics'] = model_metrics

        # Feature evaluation
        if self.use_features:
            for batch_windows, batch_labels in val_generator():
                # Extract features
                features = self.extract_features(batch_windows)

                # Evaluate handcrafted and CNN features
                feature_metrics = self.feature_evaluator.evaluate_features(
                    features['handcrafted'],
                    batch_labels
                )
                evaluation_results['feature_metrics']['handcrafted'] = feature_metrics

                feature_metrics = self.feature_evaluator.evaluate_features(
                    features['cnn'],
                    batch_labels
                )
                evaluation_results['feature_metrics']['cnn'] = feature_metrics

                # Evaluate unsupervised features
                if self.approach != 'static':
                    for name, extractor in self.extractors['unsupervised'].items():
                        # Get unsupervised metrics (reconstruction error, etc.)
                        unsupervised_metrics = self._evaluate_unsupervised_extractor(
                            extractor,
                            batch_windows,
                            name
                        )
                        evaluation_results['unsupervised_metrics'][name] = unsupervised_metrics

        return evaluation_results

    def _evaluate_unsupervised_extractor(self, extractor, windows, name):
        """Evaluate specific unsupervised extractor"""
        metrics = {}

        if isinstance(extractor, (AutoencoderExtractor, RecurrentAutoencoder)):
            reconstructed = extractor.reconstruct(windows)
            metrics['reconstruction_error'] = np.mean((windows - reconstructed) ** 2)
            features = extractor.extract(windows)
            metrics['feature_quality'] = self.feature_evaluator.compute_feature_quality(features)

        elif isinstance(extractor, DECExtractor):
            features = extractor.extract(windows)
            metrics['cluster_quality'] = self.feature_evaluator.compute_cluster_quality(features)

        elif isinstance(extractor, SOMExtractor):
            metrics['quantization_error'] = extractor.som.quantization_error(
                extractor._preprocess_window(windows)
            )
            metrics['topographic_error'] = np.mean([
                extractor._compute_topographic_error(sample)
                for sample in extractor._preprocess_window(windows)
            ])

        return metrics
