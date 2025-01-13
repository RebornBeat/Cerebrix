from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Tuple, Dict, Any

class BaseArchitecture(ABC):
    def __init__(self,
                 num_classes: int,
                 input_shape: Tuple[int, ...] = None,
                 **kwargs):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None

    @abstractmethod
    def build(self) -> tf.keras.Model:
        """Build and return the model architecture"""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        pass
