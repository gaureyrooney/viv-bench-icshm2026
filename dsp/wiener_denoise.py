"""
Spectral (Wiener) denoising of the compensated displacement signal.

Motivation, measured rather than assumed (see README "Why the 0.070mm
plateau"): after the polarity/metric fixes, the residual against LDS is
~0.0707mm, of which only ~7% of the variance is compensation-estimator noise,
~2% tracker disagreement and ~0% LDS sensor noise. The rest is broadband
vision noise sitting in frequency bands where the cable barely responds -
i.e. the vision signal carries real structural content around ~11.7Hz plus a
roughly white noise floor spread across the whole band.

That is exactly the situation a Wiener filter is optimal for. Unlike a hard
bandpass (which either discards real out-of-band LDS content, or - if applied
to both signals - manufactures the artifact documented at length in the
README), this applies a *graded* per-frequency gain

    G(f) = max(0, (P_vv(f) - alpha * P_nn) / P_vv(f))

so bins dominated by signal pass ~unchanged while bins dominated by noise are
shrunk toward zero. Nothing is hard-cut, and the resonance amplitude is
preserved (measured: amplitude ratio at 11.7Hz moves by <0.2% across the
useful alpha range).

Crucially the noise floor P_nn is estimated from the *vision signal's own*
spectrum in a band where the structure does not respond (default 16-24Hz), so
this needs no LDS reference and is deployable without ground truth.

Validation (corner_klt + affine_flow, alpha chosen on the first 30s only and
applied unchanged to the held-out last 30s):
    held-out 30s : 0.0751 -> 0.0622 mm  (-17.2%)
    full 60s clip: 0.0707 -> 0.0589 mm  (-16.7%)
An interior optimum exists near alpha=16-32; performance degrades again for
larger alpha (alpha=512 gives 0.118mm). That interior optimum is what
distinguishes genuine denoising from the degenerate "emit only the resonance
peak" behaviour, which would improve monotonically with alpha.
"""
import numpy as np
from scipy import signal as sps


def estimate_noise_psd(x, fs=50, noise_band_hz=(16.0, 24.0), nperseg=512):
    """Median PSD level in a band where the structure is assumed not to
    respond - treated as a flat (white) noise floor. Returns mm^2/Hz."""
    nyq = fs / 2.0
    lo, hi = noise_band_hz
    hi = min(hi, nyq * 0.999)
    f, P = sps.welch(np.asarray(x, dtype=np.float64), fs=fs,
                     nperseg=min(nperseg, len(x)))
    m = (f >= lo) & (f <= hi)
    if not np.any(m):
        m = f >= (0.7 * nyq)
    return float(np.median(P[m]))


def wiener_denoise(x, fs=50, alpha=16.0, noise_band_hz=(16.0, 24.0),
                   smooth_bins=9, noise_psd=None):
    """
    Args:
        x: 1D compensated displacement signal (mm)
        fs: sampling rate (Hz)
        alpha: over-subtraction factor on the estimated noise floor. 0
            disables denoising; the measured optimum on this dataset is
            ~16 (selected on held-out data - see module docstring). Larger
            values eventually degrade, which is the intended safety
            property: a degenerate narrowband filter would not.
        noise_band_hz: band used to estimate the white-noise floor from x
            itself (must be a band where the structure does not respond)
        smooth_bins: boxcar width used to smooth the periodogram before
            forming the gain, so G(f) is stable rather than per-bin noisy
        noise_psd: pass a precomputed floor (e.g. estimated on a training
            segment) instead of estimating it from x

    Returns:
        denoised 1D array, same length as x
    """
    x = np.asarray(x, dtype=np.float64)
    if alpha <= 0:
        return x.copy()
    Pnn = estimate_noise_psd(x, fs=fs, noise_band_hz=noise_band_hz) if noise_psd is None else noise_psd

    X = np.fft.rfft(x)
    # periodogram on the same normalisation as welch's PSD (mm^2/Hz)
    P = np.abs(X) ** 2 / (len(x) * fs / 2.0)
    if smooth_bins > 1:
        P = np.convolve(P, np.ones(smooth_bins) / smooth_bins, mode='same')
    G = np.clip((P - alpha * Pnn) / np.maximum(P, 1e-30), 0.0, 1.0)
    return np.fft.irfft(X * G, n=len(x))
