import tensorflow as tf
from .base import BaseFeatureExtractor
import numpy as np
from typing import List

class CNNFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__("cnn")
        self.model = self._build_model()

    def _build_model(self):
        """Build CNN model for spatial-temporal feature extraction"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(None, 16, 60)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D()
        ])
        return model

    def extract(self, window: np.ndarray) -> np.ndarray:
        """Extract CNN features from window"""
        # Reshape if needed
        if len(window.shape) == 3:
            window = window.reshape(1, *window.shape)
        return self.model(window).numpy()

    def get_feature_names(self) -> List[str]:
        return [f"cnn_feature_{i}" for i in range(64)]  # Based on final layer size
