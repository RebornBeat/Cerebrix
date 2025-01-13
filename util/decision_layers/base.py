from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Dict, Any

class BaseDecisionLayer(ABC):
    def __init__(self, num_classes: int, **kwargs):
        self.num_classes = num_classes

    @abstractmethod
    def build(self) -> tf.keras.Model:
        """Build and return the decision layer model"""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        pass
