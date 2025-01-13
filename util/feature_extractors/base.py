from abc import ABC, abstractmethod
import numpy as np
from typing import List

class BaseFeatureExtractor(ABC):
    """Base class for all feature extractors"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def extract(self, window: np.ndarray) -> np.ndarray:
        """Extract features from a window of EEG data"""
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get names of features this extractor produces"""
        pass
