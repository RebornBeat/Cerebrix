import numpy as np
import os
from threading import Lock
import logging
from datetime import datetime
from data_manager import DataManager
from queue import Queue
import mlflow
from util.model_architectures.cnn_lstm_raw import CNNLSTMRaw
from util.model_architectures.features_only_gnn import FeaturesGNN
from util.model_architectures.features_only_han import FeaturesHAN
from util.model_architectures.features_only_transformer import FeaturesTransformer
from util.model_architectures.raw_features_gnn import RawFeaturesGNN
from util.model_architectures.raw_features_han import RawFeaturesHAN
from util.model_architectures.raw_features_transformer import RawFeaturesTransformer
from util.training.integrated_trainer import IntegratedTrainer
from util.window_processor import WindowProcessor

class EEGAnalysis:
    def __init__(self, base_dir, fs=25):
        self.base_dir = base_dir
        self.data_manager = DataManager(base_dir)
        self.logs_dir = os.path.join(base_dir, "logs")
        self.fs = fs
        self.actions = set()
        self.frequency_bands = {
            'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
            'beta': (13, 30), 'gamma': (30, 60)
        }
        self.setup_logging()
        self.progress_queue = Queue()
        self.progress_lock = Lock()
        self.total_progress = 0
        self.last_progress_update = 0
        self.window_size = 5  # 200ms at 25Hz
        self.stride = 2  # 80ms at 25Hz
        self.context_size = 2
        self.max_stack_size = 3
        self.batch_size = 32
        self.feature_version = 1
        self.model_architectures = {
            'cnn_lstm_raw': CNNLSTMRaw,
            'features_gnn': FeaturesGNN,
            'features_han': FeaturesHAN,
            'features_transformer': FeaturesTransformer,
            'raw_features_gnn': RawFeaturesGNN,
            'raw_features_han': RawFeaturesHAN,
            'raw_features_transformer': RawFeaturesTransformer
        }
        self.MODEL_COMBINATIONS = {
            'raw_only': {
                'model': 'cnn_lstm_raw',
                'use_features': False,
                'use_raw': True
            },
            'features_only': {
                'transformer': {
                    'model': 'features_transformer',
                    'use_features': True,
                    'use_raw': False
                },
                'han': {
                    'model': 'features_han',
                    'use_features': True,
                    'use_raw': False
                },
                'gnn': {
                    'model': 'features_gnn',
                    'use_features': True,
                    'use_raw': False
                }
            },
            'raw_features_combined': {
                'transformer': {
                    'model': 'raw_features_transformer',
                    'use_features': True,
                    'use_raw': True
                },
                'han': {
                    'model': 'raw_features_han',
                    'use_features': True,
                    'use_raw': True
                },
                'gnn': {
                    'model': 'raw_features_gnn',
                    'use_features': True,
                    'use_raw': True
                }
            }
        }
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

    def get_data_generator(self, data_type, window_processor):
        """Create data generator with proper window processing"""
        def generator():
            for action in self.actions:
                sample_count = self.data_manager.get_sample_count(data_type, action)
                for i in range(0, sample_count, self.batch_size):
                    batch_data = []
                    batch_labels = []

                    for j in range(i, min(i + self.batch_size, sample_count)):
                        # Load raw data
                        data = self.data_manager.load_data(data_type, action, j)
                        if data is not None:
                            # Create windows
                            windows = window_processor.create_sliding_windows(data)
                            batch_data.extend(windows)
                            batch_labels.extend([self.actions.index(action)] * len(windows))

                    if batch_data:
                        yield np.array(batch_data), np.eye(len(self.actions))[batch_labels]

        return generator

    def run_analysis(self):
        try:
            self.log_and_emit("Starting model training and evaluation...")
            results = {}

            # Initialize window processor
            window_processor = WindowProcessor(
                window_size=self.window_size,
                stride=self.stride,
                context_size=self.context_size
            )

            # Train for each model combination
            for combo_type, architectures in self.MODEL_COMBINATIONS.items():
                self.log_and_emit(f"Training {combo_type} models...")

                if isinstance(architectures, dict):
                    # Multiple architectures for this combination
                    for arch_name, config in architectures.items():
                        results[f"{combo_type}_{arch_name}"] = self.train_model_combination(
                            model_type=config['model'],
                            use_features=config['use_features'],
                            use_raw=config['use_raw'],
                            window_processor=window_processor
                        )
                else:
                    # Single architecture
                    results[combo_type] = self.train_model_combination(
                        model_type=architectures['model'],
                        use_features=architectures['use_features'],
                        use_raw=architectures['use_raw'],
                        window_processor=window_processor
                    )

            return results

            self.log_and_emit("Analysis complete.")
        except Exception as e:
            self.log_and_emit(f"Error during analysis: {str(e)}", level=logging.ERROR)
            raise

    def train_model_combination(self, model_type, use_features, use_raw, window_processor):
        """Train specific model combination"""
        # Initialize correct model architecture
        results = {}

        # Get model architecture class
        if model_type not in self.model_architectures:
            raise ValueError(f"Unknown model type: {model_type}")
        model_class = self.model_architectures[model_type]

        # Train with each approach if using features
        if use_features:
            for approach in ['static', 'single_phase', 'two_phase', 'adaptive']:
                self.log_and_emit(f"Training with {approach} approach...")
                # Initialize trainer with appropriate configuration
                trainer = IntegratedTrainer(
                    model_class=model_class,
                    use_features = use_features,
                    use_hyperparameter_tuning=True,
                    use_unsupervised_updates=use_features,  # Only if using features
                    stack_specific=use_raw,  # Only for raw data processing
                    max_stack_size=self.max_stack_size if use_raw else 1,
                    approach=approach,
                    window_processor=window_processor
                )

                # Get data generators
                train_generator = self.get_data_generator('cleaned_data', window_processor)
                val_generator = self.get_data_generator('validation_data', window_processor)

                # Train model
                model = trainer.train(
                    train_generator,
                    val_generator,
                    epochs=10,
                    batch_size=self.batch_size
                )

                # Evaluate
                eval_results = trainer.evaluate(val_generator)

                results[approach] = {
                    'model': model,
                    'evaluation': eval_results
                }

        else:
            # Just train once if not using features
            trainer = IntegratedTrainer(
                model_class=model_class,
                use_hyperparameter_tuning=True,
                stack_specific=use_raw,
                max_stack_size=self.max_stack_size if use_raw else 1,
                window_processor=window_processor
            )

            train_generator = self.get_data_generator('cleaned_data', window_processor)
            val_generator = self.get_data_generator('validation_data', window_processor)

            model = trainer.train(train_generator, val_generator)
            eval_results = trainer.evaluate(val_generator)

            results['no_features'] = {
                'model': model,
                'evaluation': eval_results
            }

        return results
