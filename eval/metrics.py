"""
Evaluation metrics: RMSE, spectral analysis, phase lag.
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy import integrate


def rmse(predicted, reference):
    """
    Root mean squared error.
    
    Args:
        predicted: 1D array
        reference: 1D array (must be same length)
    
    Returns:
        rmse_value: float
    """
    residuals = predicted - reference
    return np.sqrt(np.mean(residuals ** 2))


def normalized_rmse(predicted, reference):
    """
    RMSE normalized by reference std dev (useful for comparison).
    """
    ref_std = np.std(reference)
    if ref_std < 1e-10:
        return np.inf
    return rmse(predicted, reference) / ref_std


def phase_lag_at_freq(signal_a, signal_b, target_freq_hz, fs=50):
    """
    Estimate phase lag between two signals at a specific frequency.
    
    Args:
        signal_a: reference signal (LDS)
        signal_b: test signal (vision)
        target_freq_hz: frequency of interest (Hz)
        fs: sampling rate (Hz)
    
    Returns:
        phase_lag_rad: phase lag in radians
        phase_lag_deg: phase lag in degrees
        coherence: magnitude-squared coherence at that frequency
    """
    # Cross-spectral density. Coherence needs Pxy, Pxx, Pyy estimated with the
    # *same* windowing/segmentation (Cauchy-Schwarz only bounds |Pxy|^2 <=
    # Pxx*Pyy <= 1 when all three come from a consistent estimator). Mixing
    # csd() (Hann-windowed Welch) with periodogram() (boxcar, no averaging)
    # breaks that bound and can silently produce coherence > 1.
    nperseg = min(len(signal_a), 1024)
    f, pxy = scipy_signal.csd(signal_a, signal_b, fs=fs, nperseg=nperseg)
    _, coh = scipy_signal.coherence(signal_a, signal_b, fs=fs, nperseg=nperseg)

    # Find frequency bin closest to target
    freq_idx = np.argmin(np.abs(f - target_freq_hz))

    # Phase = arg(pxy)
    phase_lag_rad = np.angle(pxy[freq_idx])
    phase_lag_deg = np.degrees(phase_lag_rad)

    coherence = float(np.clip(coh[freq_idx], 0.0, 1.0))

    return phase_lag_rad, phase_lag_deg, coherence


def amplitude_ratio_at_freq(signal_ref, signal_test, target_freq_hz, fs=50):
    """
    Ratio of amplitudes at a specific frequency.
    
    Returns:
        ratio: amplitude(signal_test) / amplitude(signal_ref)
    """
    f, pxx_ref = scipy_signal.periodogram(signal_ref, fs=fs)
    _, pxx_test = scipy_signal.periodogram(signal_test, fs=fs)
    
    freq_idx = np.argmin(np.abs(f - target_freq_hz))
    
    amp_ref = np.sqrt(pxx_ref[freq_idx])
    amp_test = np.sqrt(pxx_test[freq_idx])
    
    ratio = amp_test / (amp_ref + 1e-10)
    return ratio


def power_spectrum(signal_1d, fs=50, nperseg=None):
    """
    Compute power spectral density.
    
    Returns:
        f: frequency array (Hz)
        psd: power spectral density
    """
    if nperseg is None:
        nperseg = min(len(signal_1d), 2048)
    
    f, psd = scipy_signal.welch(signal_1d, fs=fs, nperseg=nperseg)
    return f, psd


def energy_in_band(signal_1d, freq_low_hz, freq_high_hz, fs=50):
    """
    Fraction of signal energy in frequency band [freq_low_hz, freq_high_hz].
    """
    f, psd = power_spectrum(signal_1d, fs=fs)
    mask = (f >= freq_low_hz) & (f <= freq_high_hz)
    energy_band = integrate.trapezoid(psd[mask], f[mask])
    energy_total = integrate.trapezoid(psd, f)
    return energy_band / (energy_total + 1e-10)


class MetricsReport:
    """Container for comprehensive evaluation results."""
    
    def __init__(self, predicted_mm, reference_mm, fs=50):
        """
        Compute all metrics.
        
        Args:
            predicted_mm: vision-based displacement (mm)
            reference_mm: LDS reference (mm)
            fs: sampling rate (Hz)
        """
        # Ensure same length
        min_len = min(len(predicted_mm), len(reference_mm))
        self.predicted = predicted_mm[:min_len]
        self.reference = reference_mm[:min_len]
        self.fs = fs
        
        # Basic
        self.rmse_mm = rmse(self.predicted, self.reference)
        self.normalized_rmse = normalized_rmse(self.predicted, self.reference)
        self.mae = np.mean(np.abs(self.predicted - self.reference))
        
        # Spectral at VIV peak (11.7 Hz)
        self.phase_lag_rad, self.phase_lag_deg, self.coherence_117hz = \
            phase_lag_at_freq(self.reference, self.predicted, 11.7, fs=self.fs)
        
        # Amplitude
        self.amplitude_ratio_117hz = amplitude_ratio_at_freq(self.reference, self.predicted, 11.7, fs=self.fs)
        
        # Energy distribution
        self.energy_10_25hz = energy_in_band(self.reference, 10, 25, fs=self.fs)
        self.energy_2_5hz = energy_in_band(self.reference, 2, 5, fs=self.fs)
        
    def __repr__(self):
        return f"""
MetricsReport:
  RMSE: {self.rmse_mm:.4f} mm
  Normalized RMSE: {self.normalized_rmse:.4f}
  MAE: {self.mae:.4f} mm
  Phase lag @ 11.7 Hz: {self.phase_lag_deg:.2f}°
  Coherence @ 11.7 Hz: {self.coherence_117hz:.3f}
  Amplitude ratio @ 11.7 Hz: {self.amplitude_ratio_117hz:.3f}
  Energy 10-25 Hz: {self.energy_10_25hz:.1%}
  Energy 2-5 Hz: {self.energy_2_5hz:.1%}
        """.strip()
