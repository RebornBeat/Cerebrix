from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.stats import entropy
from util.evaluation.base import BaseEvaluator
import psutil

class FeatureEvaluator(BaseEvaluator):
    def __init__(self, eval_frequency: int = 10):
        super().__init__(eval_frequency)
        self.metrics = {
            'reconstruction_error': [],
            'feature_quality': [],
            'stability': [],
            'discriminative_power': [],
            'training_time': [],
            'memory_usage': []
        }

    def evaluate_features(self, features: np.ndarray, labels: np.ndarray = None) -> Dict[str, float]:
        """Evaluate feature quality with optional labels"""
        metrics = {
            'feature_quality': self.compute_feature_quality(features),
            'stability': self.compute_stability(features),
            'memory_usage': self.get_memory_usage()
        }

        if labels is not None:
            metrics['discriminative_power'] = self.compute_discriminative_power(features, labels)

        return metrics

    def compute_feature_quality(self, features: np.ndarray) -> float:
        """Compute unsupervised feature quality metrics"""
        # Cluster the features
        kmeans = KMeans(n_clusters=min(8, len(features)))
        clusters = kmeans.fit_predict(features)

        # Compute metrics
        silhouette = silhouette_score(features, clusters)
        correlation = np.mean(np.abs(np.corrcoef(features.T)))
        feature_entropy = np.mean([entropy(feature) for feature in features.T])

        return np.mean([silhouette, 1 - correlation, feature_entropy])

    def compute_discriminative_power(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Compute how well features separate classes"""
        # Convert one-hot to class indices if needed
        if len(labels.shape) > 1:
            labels = np.argmax(labels, axis=1)

        # Compute within-class and between-class scatter
        class_means = []
        within_class_scatter = 0
        global_mean = np.mean(features, axis=0)

        for class_idx in np.unique(labels):
            class_samples = features[labels == class_idx]
            class_mean = np.mean(class_samples, axis=0)
            class_means.append(class_mean)
            within_class_scatter += np.sum((class_samples - class_mean) ** 2)

        between_class_scatter = sum(len(features[labels == i]) *
                                  np.sum((mean - global_mean) ** 2)
                                  for i, mean in enumerate(class_means))

        return between_class_scatter / (within_class_scatter + 1e-10)

    def compute_stability(self, features: np.ndarray) -> float:
        """Compute stability of features over time"""
        if len(features) < 2:
            return 1.0
        feature_diffs = np.diff(features, axis=0)
        return 1.0 - np.mean(np.std(feature_diffs, axis=0))

    def compute_cluster_quality(self, features: np.ndarray) -> float:
        """Compute clustering quality metrics"""
        kmeans = KMeans(n_clusters=min(8, len(features)))
        clusters = kmeans.fit_predict(features)

        # Compute cluster metrics
        silhouette = silhouette_score(features, clusters)
        inertia = kmeans.inertia_

        # Normalize inertia
        normalized_inertia = 1.0 / (1.0 + inertia)

        return np.mean([silhouette, normalized_inertia])

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
