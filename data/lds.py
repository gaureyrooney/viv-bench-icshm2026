"""
LDS data loader: 10 kHz → 50 Hz with anti-alias filtering.
"""
import numpy as np
import pandas as pd
from scipy import signal


def load_lds_raw(xlsx_path):
    """
    Load raw LDS displacement from Excel file.
    Returns: (samples_array, fs_hz) where samples_array is 1D
    """
    df = pd.read_excel(xlsx_path)
    # Assume single column 'LDS'
    lds = df.iloc[:, 0].values.astype(np.float64)
    fs_original = 10000  # Hz
    return lds, fs_original


def decimate_lds(lds_raw, fs_from=10000, fs_to=50, cutoff_hz=25):
    """
    Anti-alias filter (Butterworth LP) and decimate LDS from fs_from to fs_to.
    
    Args:
        lds_raw: 1D array, raw LDS data
        fs_from: original sampling rate (Hz)
        fs_to: target sampling rate (Hz)
        cutoff_hz: LP cutoff frequency (Hz), should be < fs_to/2
    
    Returns:
        lds_decimated: 1D array at fs_to Hz
        decimation_factor: N where fs_to = fs_from / N
    """
    # Design LP filter: cutoff at cutoff_hz, order 4 (steep but stable)
    nyquist_from = fs_from / 2
    normalized_cutoff = cutoff_hz / nyquist_from
    
    # sosfiltfilt (zero-phase, forward-backward), not sosfilt: a forward-only
    # IIR filter has real group delay (~18ms/-73deg at 11.7Hz for this
    # cutoff/order - confirmed empirically). This used to regress RMSE when
    # combined with align_signals' old single-pass correlation search (it
    # shifted the detected lag onto a worse-correlating, aliased peak) - now
    # safe now that align_signals uses find_time_offset_robust's coarse-then
    # -fine search, which is immune to exactly that failure mode. See README
    # "Per-frame diff" / "align_signals robustness".
    sos = signal.butter(4, normalized_cutoff, btype='low', output='sos')
    lds_filtered = signal.sosfiltfilt(sos, lds_raw)
    
    # Decimate
    decimation_factor = fs_from // fs_to
    lds_decimated = lds_filtered[::decimation_factor]
    
    return lds_decimated, decimation_factor


def load_lds_prepared(xlsx_path, fs_target=50, cutoff_hz=25):
    """
    Convenience function: load raw LDS, filter, decimate to fs_target.
    
    Returns:
        lds_50hz: 1D array at 50 Hz
        time_s: time vector in seconds
        meta: dict with metadata
    """
    lds_raw, fs_raw = load_lds_raw(xlsx_path)
    lds_50hz, decimation = decimate_lds(lds_raw, fs_from=fs_raw, fs_to=fs_target, cutoff_hz=cutoff_hz)
    
    time_s = np.arange(len(lds_50hz)) / fs_target
    
    meta = {
        'fs_original': fs_raw,
        'fs_final': fs_target,
        'decimation_factor': decimation,
        'cutoff_hz': cutoff_hz,
        'n_samples_original': len(lds_raw),
        'n_samples_final': len(lds_50hz),
        'duration_s': time_s[-1],
    }
    
    return lds_50hz, time_s, meta
