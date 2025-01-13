import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class IntentResult:
    """Stores results of intent detection"""
    predicted_action: str
    commitment_level: float
    timestamp: float
    confidence: float

class IntentDetector:
    def __init__(self,
                 model_path: str,
                 actions: List[str],
                 window_processor,
                 frequency_bands: Dict[str, Tuple[int, int]],
                 max_latency: int = 200,
                 commitment_threshold: float = 0.6):
        """
        Initialize Intent Detector

        Args:
            model_path: Path to saved model
            actions: List of possible actions
            window_processor: WindowProcessor instance for processing EEG data
            frequency_bands: Dictionary mapping band names to frequency ranges
            max_latency: Maximum allowable latency in ms
            commitment_threshold: Threshold for commitment detection
        """
        self.model = tf.keras.models.load_model(model_path)
        self.actions = actions
        self.window_processor = window_processor
        self.frequency_bands = frequency_bands
        self.max_latency = max_latency
        self.commitment_threshold = commitment_threshold

    def detect_intent(self, live_eeg_data: np.ndarray) -> List[IntentResult]:
        """
        Detect intent from live EEG data

        Args:
            live_eeg_data: Raw EEG data of shape (time, channels, frequencies)

        Returns:
            List of IntentResult objects containing predictions and metadata
        """
        results = []
        windows = self.window_processor.create_sliding_windows(live_eeg_data)

        for i, window in enumerate(windows):
            # Make prediction
            prediction = self.model.predict(window[np.newaxis, ...])
            predicted_action = self.actions[np.argmax(prediction[0])]
            confidence = np.max(prediction[0])

            # Analyze commitment
            commitment = self.analyze_commitment(window)

            # Calculate timestamp
            timestamp = i * self.window_processor.stride / self.window_processor.fs

            # Check latency threshold
            if timestamp * 1000 > self.max_latency:
                break

            results.append(IntentResult(
                predicted_action=predicted_action,
                commitment_level=commitment,
                timestamp=timestamp,
                confidence=confidence
            ))

        return results

    def analyze_commitment(self, eeg_data: np.ndarray) -> float:
        """
        Analyze commitment level from EEG data

        Args:
            eeg_data: Single window of EEG data

        Returns:
            Commitment score between 0 and 1
        """
        # Calculate band powers
        alpha_power = np.mean(self._compute_band_power(eeg_data, *self.frequency_bands['alpha']))
        beta_power = np.mean(self._compute_band_power(eeg_data, *self.frequency_bands['beta']))
        theta_power = np.mean(self._compute_band_power(eeg_data, *self.frequency_bands['theta']))

        # Calculate commitment score
        commitment = beta_power / (alpha_power + theta_power)

        return commitment

    def _compute_band_power(self, data: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """Compute power in specific frequency band"""
        freq_bins = np.arange(data.shape[-1])  # Assume last dim is frequencies
        idx = np.logical_and(freq_bins >= low_freq, freq_bins < high_freq)
        return np.mean(data[..., idx], axis=-1)
