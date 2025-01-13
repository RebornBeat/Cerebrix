from .base import BaseFeatureExtractor
import numpy as np
from scipy import signal
from typing import Dict, List

class HandcraftedFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__("handcrafted")
        self.frequency_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 60)
        }

    def extract(self, window: np.ndarray) -> np.ndarray:
        """Extract handcrafted features from window"""
        features = []

        # Compute band powers first as they're used in multiple features
        band_powers = {}
        for band_name, (low, high) in self.frequency_bands.items():
            band_powers[band_name] = self._compute_band_power(window, low, high)

        # Frequency domain features
        features.extend(self._extract_spectral_features(window, band_powers))

        # Temporal features
        features.extend(self._extract_temporal_features(window, band_powers))

        # Spatial features
        features.extend(self._extract_spatial_features(window))

        # Connectivity features
        features.extend(self._compute_connectivity_features(window))

        return np.array(features)

    def _extract_spectral_features(self, window: np.ndarray, band_powers: Dict) -> List[float]:
        features = []

        # Band powers
        for band_power in band_powers.values():
            features.extend([
                np.mean(band_power),
                np.std(band_power),
                np.max(band_power),
                np.min(band_power),
                np.median(band_power),
                np.ptp(band_power),
                np.mean(np.diff(band_power))
            ])

        # Spectral ratios
        delta = band_powers['delta']
        theta = band_powers['theta']
        alpha = band_powers['alpha']
        beta = band_powers['beta']
        gamma = band_powers['gamma']

        features.extend([
            np.mean(theta/alpha),
            np.mean(beta/(alpha + theta)),
            np.mean((beta + gamma) / (delta + theta)),
        ])

        # Temporal stability for each band
        for band in band_powers.values():
            features.append(np.mean(np.abs(np.diff(band))))

        # Spectral entropy
        spectral_entropy = -np.sum(window * np.log2(window + 1e-10), axis=2).mean(axis=1)
        features.extend(spectral_entropy)

        # Commitment feature
        commitment = np.mean(beta) / (np.mean(alpha) + np.mean(theta))
        features.append(commitment)

        return features

    def _extract_temporal_features(self, window: np.ndarray, band_powers: Dict) -> List[float]:
        features = []

        # Time-domain features (adapted for frequency domain data)
        # Hjorth parameters
        hjorth_activity = np.var(window, axis=(0, 2)).mean()
        hjorth_mobility = np.sqrt(np.var(np.diff(window, axis=0), axis=(0, 2)) /
                                np.var(window, axis=(0, 2))).mean()
        hjorth_complexity = (np.sqrt(np.var(np.diff(np.diff(window, axis=0), axis=0), axis=(0, 2)) /
                           np.var(np.diff(window, axis=0), axis=(0, 2))) / hjorth_mobility).mean()

        features.extend([hjorth_activity, hjorth_mobility, hjorth_complexity])

        # Wavelet features (adapted for frequency domain data)
        wavelet_features = [np.mean(np.abs(window[:, :, i:i+8]))
                          for i in range(0, window.shape[2], 8)]
        features.extend(wavelet_features)

        # Statistical features
        features.extend([
            np.mean(np.abs(np.diff(window, axis=0))),  # Mean absolute difference
            np.std(window),                            # Standard deviation
            self.peak_to_peak(window, axis=0).mean() # Peak-to-peak amplitude
        ])

        return features

    def _extract_spatial_features(self, window: np.ndarray) -> List[float]:
        features = []

        # Channel correlations
        corr_matrix = np.corrcoef(window.reshape(-1, window.shape[-1]))
        features.extend([
            np.mean(corr_matrix),
            np.std(corr_matrix),
            np.max(corr_matrix)
        ])

        # Spatial complexity
        for i in range(window.shape[1]):  # For each channel
            other_channels = np.delete(window, i, axis=1)
            features.append(np.mean(np.corrcoef(window[:, i], other_channels.mean(axis=1))))

        return features

    def _compute_band_power(self, data: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """Compute power in specific frequency band"""
        freq_bins = np.arange(60)  # 0-59 Hz
        idx = np.logical_and(freq_bins >= low_freq, freq_bins < high_freq)
        return np.mean(data[:, :, idx], axis=2)

    def _compute_connectivity_features(self, data: np.ndarray) -> np.ndarray:
        """Compute connectivity features across channels and frequency bins"""
        n_channels = data.shape[1]
        n_freq_bins = data.shape[2]
        connectivity = np.zeros((n_channels, n_channels, n_freq_bins))

        for f in range(n_freq_bins):
            freq_data = data[:, :, f]
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    connectivity[i, j, f] = np.abs(np.corrcoef(freq_data[:, i], freq_data[:, j])[0, 1])
                    connectivity[j, i, f] = connectivity[i, j, f]

        # Compute summary statistics
        mean_connectivity = np.mean(connectivity, axis=(0, 1))
        max_connectivity = np.max(connectivity, axis=(0, 1))
        std_connectivity = np.std(connectivity, axis=(0, 1))

        return np.concatenate([mean_connectivity, max_connectivity, std_connectivity])

    def peak_to_peak(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """Compute peak-to-peak amplitude"""
        return np.ptp(data, axis=axis)

def get_feature_names(self) -> List[str]:
    names = []

    # Store expected shapes
    n_channels = 16  # Standard number of channels
    n_frequencies = 60  # Standard number of frequencies

    # Spectral feature names
    for band in self.frequency_bands:
        names.extend([
            f"{band}_mean",
            f"{band}_std",
            f"{band}_max",
            f"{band}_min",
            f"{band}_median",
            f"{band}_ptp",
            f"{band}_diff_mean"
        ])

    # Band ratio names
    names.extend([
        "theta_alpha_ratio",
        "beta_alpha_theta_ratio",
        "beta_gamma_delta_theta_ratio"
    ])

    # Temporal stability names
    for band in self.frequency_bands:
        names.append(f"{band}_temporal_stability")

    # Spectral entropy names
    names.append("spectral_entropy")

    # Commitment feature
    names.append("commitment")

    # Temporal feature names
    names.extend([
        "hjorth_activity",
        "hjorth_mobility",
        "hjorth_complexity",
        "mean_abs_diff",
        "std_dev",
        "peak_to_peak"
    ])

    # Wavelet feature names
    n_wavelet_features = n_frequencies // 8
    names.extend([f"wavelet_feature_{i}" for i in range(n_wavelet_features)])

    # Spatial feature names
    names.extend([
        "mean_correlation",
        "std_correlation",
        "max_correlation"
    ])

    # Spatial complexity names
    names.extend([f"channel_{i}_spatial_complexity" for i in range(n_channels)])

    # Connectivity feature names
    names.extend([
        f"mean_connectivity_{i}" for i in range(n_frequencies)
    ])
    names.extend([
        f"max_connectivity_{i}" for i in range(n_frequencies)
    ])
    names.extend([
        f"std_connectivity_{i}" for i in range(n_frequencies)
    ])

    return names
