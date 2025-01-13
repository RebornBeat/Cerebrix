import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import defaultdict
from util.evaluation.base import BaseEvaluator
import time

class ModelEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.metrics = {
            'accuracy': tf.keras.metrics.Accuracy(),
            'precision': tf.keras.metrics.Precision(),
            'recall': tf.keras.metrics.Recall(),
            'auc': tf.keras.metrics.AUC(),
            'confusion_matrix': None,
            'cross_val_scores': [],
            'inference_time': [],
            'memory_usage': []
        }

    def evaluate_epoch(self, models, val_generator, max_stack_size=None):
        """Evaluate models for one epoch"""
        metrics = defaultdict(float)
        num_batches = 0

        for batch_data, batch_labels in val_generator():
            if max_stack_size:
                batch_metrics = self._evaluate_stack_specific(
                    models, batch_data, batch_labels, max_stack_size)
            else:
                batch_metrics = self._evaluate_single_model(
                    models[1], batch_data, batch_labels)

            for k, v in batch_metrics.items():
                metrics[k] += v
            num_batches += 1

        return {k: v/num_batches for k, v in metrics.items()}

    def _evaluate_stack_specific(self, models, batch_data, batch_labels, max_stack_size):
        """Evaluate stack-specific models"""
        metrics = {}
        for stack_size in range(1, max_stack_size + 1):
            predictions = models[stack_size].predict(batch_data)
            metrics[f'stack_{stack_size}_accuracy'] = accuracy_score(
                np.argmax(batch_labels, axis=1),
                np.argmax(predictions, axis=1)
            )
        return metrics

    def _evaluate_single_model(self, model, batch_data, batch_labels):
        """Evaluate single model"""
        predictions = model.predict(batch_data)
        return {
            'accuracy': accuracy_score(
                np.argmax(batch_labels, axis=1),
                np.argmax(predictions, axis=1)
            )
        }

    def evaluate_model(self, model, data, labels):
        start_time = time.time()
        predictions = model.predict(data)
        inference_time = time.time() - start_time

        metrics = {
            'accuracy': accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1)),
            'confusion_matrix': confusion_matrix(np.argmax(labels, axis=1), np.argmax(predictions, axis=1)),
            'classification_report': classification_report(np.argmax(labels, axis=1),
                                                        np.argmax(predictions, axis=1)),
            'inference_time': inference_time,
            'memory_usage': self.get_memory_usage()
        }

        return metrics
