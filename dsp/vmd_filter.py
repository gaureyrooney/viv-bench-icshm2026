"""
Variational Mode Decomposition (VMD) residual cleanup.

Two-stage UAV motion correction (see e.g. Sun et al., "A two-stage correction
method for UAV movement-induced errors", 2024): stage 1 is a geometric
transform from stationary reference features (our AffineFlowCompensator);
stage 2 adaptively decomposes the *residual* signal and discards components
that aren't the known structural response frequency. VMD (Dragomiretskiy &
Zosso, 2014) decomposes a signal into K band-limited "intrinsic mode
functions" (IMFs), each with its own center frequency - instead of removing
everything outside a fixed filter band (bandpass), we keep only the mode(s)
whose fitted center frequency actually lands near the known VIV frequency,
which adapts to the signal rather than assuming a band a priori.
"""
import numpy as np
from vmdpy import VMD


def vmd_denoise(signal_1d, fs=50, K=5, target_freq_hz=11.7, freq_tol_hz=3.0,
                 alpha=2000, tau=0.0, tol=1e-7):
    """
    Decompose signal_1d into K modes via VMD and reconstruct using only the
    mode(s) whose converged center frequency is within freq_tol_hz of
    target_freq_hz (falls back to the single closest mode if none qualify).

    Args:
        signal_1d: 1D array (e.g. compensated vision displacement, mm)
        fs: sampling rate (Hz)
        K: number of modes to decompose into. Too few merges the structural
           response with noise/drift into one mode; too many can split the
           true response across multiple near-duplicate modes (observed
           empirically here with K=3 on a synthetic two-tone test - not
           harmful for reconstruction since duplicates get kept together,
           but wasteful). K=5 is a reasonable default for a single dominant
           resonance plus low-freq drift plus broadband noise.
        target_freq_hz: known structural response frequency to keep
        freq_tol_hz: how close a mode's center frequency must be to count
        alpha, tau, tol: VMD optimization parameters (bandwidth constraint,
            noise-tolerance, convergence tolerance) - see Dragomiretskiy &
            Zosso (2014) / vmdpy defaults.

    Returns:
        reconstructed: 1D array, same length as signal_1d (VMD's output can
            be 1 sample shorter; the last value is repeated to pad back)
        mode_freqs_hz: (K,) converged center frequency of each mode
        keep_mask: (K,) bool, which modes were kept
    """
    n = len(signal_1d)
    u, u_hat, omega = VMD(signal_1d, alpha, tau, K, 0, 1, tol)
    mode_freqs_hz = omega[-1] * fs

    keep_mask = np.abs(mode_freqs_hz - target_freq_hz) <= freq_tol_hz
    if not keep_mask.any():
        keep_mask = np.zeros(K, dtype=bool)
        keep_mask[int(np.argmin(np.abs(mode_freqs_hz - target_freq_hz)))] = True

    reconstructed = u[keep_mask].sum(axis=0)
    if len(reconstructed) < n:
        reconstructed = np.concatenate([reconstructed, np.full(n - len(reconstructed), reconstructed[-1])])
    elif len(reconstructed) > n:
        reconstructed = reconstructed[:n]

    return reconstructed, mode_freqs_hz, keep_mask
