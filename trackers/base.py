"""
Base tracker interface and utilities.
"""
import numpy as np
from abc import ABC, abstractmethod


class Tracker(ABC):
    """
    Abstract base class for all trackers.
    
    A tracker takes a stack of frames and a set of query points (pixel coordinates),
    and returns estimated trajectories for those points over time.
    """
    
    def __init__(self, name, max_frames=None):
        """
        Args:
            name: human-readable tracker name
            max_frames: if set, only use first max_frames from input
        """
        self.name = name
        self.max_frames = max_frames
    
    @abstractmethod
    def track(self, frames, query_points):
        """
        Track a set of query points through a stack of frames.
        
        Args:
            frames: (T, H, W) or (T, H, W, 3) array of video frames (uint8)
            query_points: (N, 2) array of (x, y) pixel coordinates to track
        
        Returns:
            trajectories: (T, N, 2) array of (x, y) positions over time
            confidence: (T, N) array of confidence scores [0, 1]
            meta: dict with tracker-specific metadata
        """
        pass
    
    def __repr__(self):
        return f"{self.name} (max_frames={self.max_frames})"


def generate_synthetic_motion(frame_base, displacement_px_1d, projection_angle=0):
    """
    Generate synthetic video frames by shifting the base frame.
    
    Args:
        frame_base: (H, W) or (H, W, 3) base frame
        displacement_px_1d: (T,) 1D displacement in pixels (along projection_angle)
        projection_angle: angle of motion direction (radians)
    
    Returns:
        frames: (T, H, W) or (T, H, W, 3) shifted frames
    """
    T = len(displacement_px_1d)
    
    # Create output
    if frame_base.ndim == 3:
        frames = np.zeros((T, frame_base.shape[0], frame_base.shape[1], frame_base.shape[2]), dtype=np.uint8)
    else:
        frames = np.zeros((T, frame_base.shape[0], frame_base.shape[1]), dtype=np.uint8)
    
    # Compute 2D shifts
    shift_x = displacement_px_1d * np.cos(projection_angle)
    shift_y = displacement_px_1d * np.sin(projection_angle)
    
    for t in range(T):
        # Sub-pixel shift using bilinear interpolation
        tx, ty = shift_x[t], shift_y[t]
        
        from scipy import ndimage
        if frame_base.ndim == 3:
            # Color frame
            shifted = ndimage.shift(frame_base, (ty, tx, 0), order=1, mode='constant', cval=0)
        else:
            # Grayscale
            shifted = ndimage.shift(frame_base, (ty, tx), order=1, mode='constant', cval=0)
        
        frames[t] = np.clip(shifted, 0, 255).astype(np.uint8)
    
    return frames


def synthetic_validation_test(test_amplitude_px=0.5, test_frequency_hz=2.0, duration_s=10, fs=50):
    """
    Generate synthetic sine motion and validate the harness can recover it.
    
    Returns:
        frames: (T, H, W) synthetic video
        true_displacement_px: (T,) ground truth motion
        time_s: (T,) time vector
    """
    # Create a simple test frame: checkerboard pattern
    H, W = 256, 256
    square_size = 32
    frame_base = np.tile(
        np.array([[255, 0], [0, 255]], dtype=np.uint8),
        (H // square_size // 2, W // square_size // 2)
    )
    frame_base = np.repeat(np.repeat(frame_base, square_size, axis=0), square_size, axis=1)
    frame_base = frame_base[:H, :W]  # Crop to exact size
    
    # Generate sine motion
    T = int(duration_s * fs)
    time_s = np.arange(T) / fs
    true_displacement_px = test_amplitude_px * np.sin(2 * np.pi * test_frequency_hz * time_s)
    
    # Create synthetic video
    frames = generate_synthetic_motion(frame_base, true_displacement_px, projection_angle=0)
    
    return frames, true_displacement_px, time_s
