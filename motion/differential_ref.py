"""
Differential reference ego-motion compensation.

Track background reference points and target separately, then subtract
background motion to remove camera drift from target.
"""
import numpy as np


def differential_reference_compensation(trajectories_target, trajectories_background):
    """
    Remove camera motion by subtracting background motion from target.
    
    Args:
        trajectories_target: (T, N_target, 2) target trajectories
        trajectories_background: (T, N_bg, 2) background reference trajectories
    
    Returns:
        trajectories_compensated: (T, N_target, 2) drift-removed target
        bg_motion: (T, 2) mean background motion per frame
    """
    T, N_target, _ = trajectories_target.shape
    
    # Compute mean background motion per frame
    bg_motion = np.mean(trajectories_background, axis=1)  # (T, 2)
    
    # Subtract from target
    trajectories_comp = trajectories_target.copy()
    for t in range(T):
        trajectories_comp[t, :, :] -= bg_motion[t, :]
    
    return trajectories_comp, bg_motion


def extract_relative_motion(trajectories):
    """
    Compute relative motion (displacement from first frame).
    
    Args:
        trajectories: (T, N, 2)
    
    Returns:
        relative_motion: (T, N, 2) with first frame at zero
    """
    return trajectories - trajectories[0:1, :, :]


def compute_motion_statistics(trajectories, axis=1):
    """
    Compute motion statistics along specified axis.
    
    Args:
        trajectories: (T, N, 2)
        axis: 0 = temporal, 1 = spatial (across points)
    
    Returns:
        dict with RMS, std, mean per frame or point
    """
    if axis == 1:  # Spatial: across N points
        rms_per_frame = np.sqrt(np.mean(trajectories**2, axis=(1, 2)))
        std_per_frame = np.std(trajectories, axis=1)  # (T, 2)
        return {
            'rms': rms_per_frame,
            'std_x': std_per_frame[:, 0],
            'std_y': std_per_frame[:, 1],
        }
    elif axis == 0:  # Temporal: over T frames
        rms_per_point = np.sqrt(np.mean(trajectories**2, axis=0))  # (N, 2)
        return {
            'rms': rms_per_point,
        }


class DifferentialReferenceCompensator:
    """
    Simple differential reference ego-motion compensation.
    """
    
    def __init__(self):
        self.bg_motion = None
    
    def fit(self, trajectories_target, trajectories_background):
        """
        Learn background motion.
        
        Args:
            trajectories_target: (T, N_target, 2)
            trajectories_background: (T, N_bg, 2)
        """
        self.bg_motion = np.mean(trajectories_background, axis=1)
    
    def compensate(self, trajectories_target):
        """Apply learned compensation."""
        if self.bg_motion is None:
            raise RuntimeError("Must call fit() first")
        
        T = trajectories_target.shape[0]
        trajectories_comp = trajectories_target.copy()
        for t in range(min(T, len(self.bg_motion))):
            trajectories_comp[t, :, :] -= self.bg_motion[t, :]
        
        return trajectories_comp
    
    def get_background_motion(self):
        """Return estimated background motion."""
        return self.bg_motion


class NoCompensator:
    """
    Identity compensator: passes target trajectories through unchanged.
    Useful as a reference point to quantify how much a real compensator helps.
    """

    def fit(self, trajectories_target, trajectories_background):
        pass

    def compensate(self, trajectories_target):
        return trajectories_target.copy()

    def get_background_motion(self):
        return None
