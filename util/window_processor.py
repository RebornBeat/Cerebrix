import numpy as np
from typing import Tuple, List, Optional, Dict, Union
import logging
from dataclasses import dataclass

@dataclass
class WindowInfo:
    """Stores metadata about windows"""
    original_shape: Tuple[int, ...]
    n_windows: int
    temporal_positions: np.ndarray  # Tracks frame positions in original data
    padding_mask: np.ndarray       # Tracks which frames are padded

class WindowProcessor:
    def __init__(self,
                 window_size: int = 5,
                 stride: int = 2,
                 context_size: int = 2,
                 debug: bool = False):
        """Initialize window processor for EEG data.

        Args:
            window_size: Number of frames per window (default 5 = 200ms at 25Hz)
            stride: Step size between windows (default 2 = 80ms at 25Hz)
            context_size: Number of past/future frames for context (default 2)
            debug: Enable debug logging
        """
        self.window_size = window_size
        self.stride = stride
        self.context_size = context_size
        self.debug = debug

        # Setup logging
        self.logger = logging.getLogger(__name__)
        log_level = logging.DEBUG if debug else logging.INFO
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def create_sliding_windows(self,
                             data: np.ndarray,
                             return_info: bool = False) -> Union[List[np.ndarray], Tuple[List[np.ndarray], WindowInfo]]:
        """Create sliding windows with dynamic padding based on frame availability.

        Args:
            data: numpy array of shape (n_frames, n_channels, n_frequencies)
            return_info: If True, returns WindowInfo object with metadata

        Returns:
            windows: List of windowed data, each with shape (2*context_size + 1, n_channels, n_frequencies)
            window_info: (optional) WindowInfo object with metadata about the windows
        """
        self.validate_input(data)

        n_frames, n_channels, n_frequencies = data.shape
        frame_shape = (n_channels, n_frequencies)
        windows = []

        # Track temporal positions and padding for metadata
        temporal_positions = []
        padding_mask = []

        for current_idx in range(0, n_frames, self.stride):
            window_frames = []
            current_temporal_positions = []
            current_padding_mask = []

            # Handle past frames
            for past_offset in range(self.context_size, 0, -1):
                past_idx = current_idx - past_offset
                if past_idx >= 0:
                    # Past frame exists
                    window_frames.append(data[past_idx])
                    current_temporal_positions.append(past_idx)
                    current_padding_mask.append(False)
                else:
                    # No past frame available, add zero padding
                    window_frames.append(np.zeros(frame_shape))
                    current_temporal_positions.append(-1)  # -1 indicates padding
                    current_padding_mask.append(True)

            # Add current frame
            window_frames.append(data[current_idx])
            current_temporal_positions.append(current_idx)
            current_padding_mask.append(False)

            # Handle future frames
            for future_offset in range(1, self.context_size + 1):
                future_idx = current_idx + future_offset
                if future_idx < n_frames:
                    # Future frame exists
                    window_frames.append(data[future_idx])
                    current_temporal_positions.append(future_idx)
                    current_padding_mask.append(False)
                else:
                    # No future frame available, add zero padding
                    window_frames.append(np.zeros(frame_shape))
                    current_temporal_positions.append(-1)  # -1 indicates padding
                    current_padding_mask.append(True)

            # Stack frames into a single window
            window = np.stack(window_frames, axis=0)
            windows.append(window)

            temporal_positions.append(current_temporal_positions)
            padding_mask.append(current_padding_mask)

            if self.debug:
                self.debug_print(f"Window {len(windows)}: Shape {window.shape}")
                self.debug_print(f"  - Past frames: {max(0, current_idx)}")
                self.debug_print(f"  - Future frames: {min(self.context_size, n_frames - current_idx - 1)}")

        if return_info:
            window_info = WindowInfo(
                original_shape=data.shape,
                n_windows=len(windows),
                temporal_positions=np.array(temporal_positions),
                padding_mask=np.array(padding_mask)
            )
            return windows, window_info

        return windows

    def debug_print(self, message: str) -> None:
        """Print debug message if debug mode is enabled."""
        if self.debug:
            self.logger.debug(message)

    def create_stacked_windows(self,
                             windows: np.ndarray,
                             stack_size: int,
                             window_info: Optional[WindowInfo] = None) -> np.ndarray:
        """Stack windows along frequency dimension while preserving temporal relationships.

        Args:
            windows: Windowed data from create_sliding_windows
            stack_size: Number of windows to stack
            window_info: Optional WindowInfo object for validation

        Returns:
            stacked_windows: Stacked windows with preserved temporal relationships
        """
        self.validate_windows(windows)
        if stack_size < 1:
            raise ValueError("stack_size must be >= 1")

        n_windows, window_length, n_channels, n_frequencies = windows.shape

        # Calculate number of possible stacks
        n_stacks = n_windows - stack_size + 1

        if n_stacks < 1:
            raise ValueError(f"Not enough windows ({n_windows}) for requested stack_size ({stack_size})")

        # Initialize stacked windows array
        stacked_windows = np.zeros((n_stacks, window_length, n_channels, n_frequencies * stack_size))

        # Create stacks
        for i in range(n_stacks):
            # Concatenate windows along frequency dimension
            stack = windows[i:i + stack_size]
            stacked_windows[i] = np.concatenate(
                [stack[j] for j in range(stack_size)],
                axis=-1  # Concatenate along frequency dimension
            )

        self.logger.debug(f"Created {n_stacks} stacks with shape {stacked_windows.shape}")

        return stacked_windows

    def validate_input(self, data: np.ndarray) -> None:
        """Validate input data shape and values."""
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a numpy array")

        if len(data.shape) != 3:
            raise ValueError(f"Expected 3D input (frames, channels, frequencies), got shape {data.shape}")

        n_frames, n_channels, n_frequencies = data.shape

        if n_frames < self.window_size:
            raise ValueError(f"Input has {n_frames} frames, need at least {self.window_size}")

        self.logger.debug(f"Validated input shape: {data.shape}")

    def validate_windows(self, windows: np.ndarray) -> None:
        """Validate windowed data shape and values."""
        if not isinstance(windows, np.ndarray):
            raise TypeError("Windows must be a numpy array")

        if len(windows.shape) != 4:
            raise ValueError(f"Expected 4D windows (n_windows, window_length, channels, frequencies), got shape {windows.shape}")

        expected_window_length = self.window_size + 2 * self.context_size
        if windows.shape[1] != expected_window_length:
            raise ValueError(f"Window length should be {expected_window_length}, got {windows.shape[1]}")

        self.logger.debug(f"Validated windows shape: {windows.shape}")

    def get_window_metadata(self,
                          window_info: WindowInfo,
                          window_idx: int) -> Dict:
        """Get metadata about a specific window."""
        if not 0 <= window_idx < window_info.n_windows:
            raise ValueError(f"Window index {window_idx} out of range [0, {window_info.n_windows})")

        return {
            'temporal_position': window_info.temporal_positions[window_idx],
            'is_padded': window_info.padding_mask[window_idx].any(),
            'padding_locations': np.where(window_info.padding_mask[window_idx])[0],
            'context_frames': np.where(window_info.context_mask[window_idx])[0]
        }

    def __repr__(self) -> str:
        return (f"WindowProcessor(window_size={self.window_size}, "
                f"stride={self.stride}, context_size={self.context_size})")
