"""
Signal processing: filtering, alignment, mean-subtraction, projection.
"""
import os

import numpy as np
from scipy import signal, ndimage


def zero_phase_filter(data, fs, cutoff_hz=1.0, order=4, btype='high'):
    """
    Zero-phase (forward-backward) filter using filtfilt. Eliminates phase
    distortion, which matters here since RMSE is compared sample-by-sample
    against LDS, not just spectrally.

    Args:
        data: 1D array
        fs: sampling rate (Hz)
        cutoff_hz: cutoff frequency (Hz) for 'low'/'high', or a (low, high)
            tuple in Hz for 'band'/'bandstop'
        order: filter order
        btype: 'high', 'low', 'band', or 'bandstop'

    Returns:
        filtered: 1D array, same length, zero-phase filtered
    """
    nyquist = fs / 2
    if btype in ('band', 'bandstop'):
        normalized_cutoff = [c / nyquist for c in cutoff_hz]
    else:
        normalized_cutoff = cutoff_hz / nyquist

    sos = signal.butter(order, normalized_cutoff, btype=btype, output='sos')
    return signal.sosfiltfilt(sos, data)


def _windowed_pearson_corr(a_full, b_full, lag):
    """
    Pearson correlation between a_full and b_full shifted by `lag` samples
    (b delayed by lag relative to a, matching find_time_offset_by_xcorr's
    convention), computed on the overlapping window only, with mean removed
    from *that window* - not from the full, unshifted signal.
    """
    n = len(a_full)
    if lag >= 0:
        a_ov = a_full[lag:]
        b_ov = b_full[:n - lag] if lag > 0 else b_full
    else:
        a_ov = a_full[:n + lag]
        b_ov = b_full[-lag:]
    if len(a_ov) < 20:
        return -np.inf
    a0 = a_ov - a_ov.mean()
    b0 = b_ov - b_ov.mean()
    denom = np.sqrt(np.sum(a0 ** 2) * np.sum(b0 ** 2))
    if denom < 1e-12:
        return -np.inf
    return float(np.dot(a0, b0) / denom)


def find_time_offset_by_xcorr(signal_a, signal_b, max_lag_samples, fs=50, subsample=True,
                               center_lag_samples=0.0, search_radius_samples=None):
    """
    Find the time offset that maximizes Pearson correlation between two
    signals, searching every integer lag in the window directly (not a single
    global cross-correlation statistic).

    This matters specifically because the VIV signal is narrowband (~85% of
    its energy sits in a ~15 Hz band around 11.7 Hz): a single global
    cross-correlation computed once against the whole (fixed-normalization)
    signal can have several comparably tall peaks spaced about one VIV period
    (1/11.7 Hz =~ 85 ms) apart, and picking the tallest one is not the same as
    picking the lag that truly maximizes the correlation of the *overlapping*
    window - two tracker runs on the same clip were observed to land on lags
    differing by ~83 ms, i.e. one aliased period, which silently wrecked
    downstream RMSE despite good narrowband coherence. Recomputing Pearson
    correlation per-candidate-lag on the overlap only avoids that.

    Still not fully immune on its own though: a blind, wide, unconstrained
    search over this landscape can pick whichever aliased peak happens to be
    tallest by a hair, which isn't stable to small changes upstream (e.g. a
    small reference-signal filtering fix shifted the true optimum by under a
    sample but flipped which peak was globally tallest, regressing RMSE - see
    README "align_signals robustness"). find_time_offset_robust() below
    layers a coarse, alias-immune stage on top of this one to fix that; this
    function itself stays a plain, honest "search this window" primitive.

    Args:
        signal_a: reference signal (LDS), 1D array
        signal_b: test signal (vision), 1D array
        max_lag_samples: half-width of the default search window (samples),
            used when search_radius_samples is not given
        fs: sampling rate (Hz) for converting lag to seconds
        subsample: if True, refine the integer-sample lag with parabolic
            interpolation of the correlation values around the best lag.
        center_lag_samples: search is centered on this lag instead of zero
            (public convention: positive = signal_b delayed). Lets a caller
            constrain the search to a neighborhood of a previously-found
            estimate instead of the full range around zero.
        search_radius_samples: half-width of the search window around
            center_lag_samples. Defaults to max_lag_samples (i.e. the
            original ±max_lag_samples-around-zero behavior when
            center_lag_samples is also left at 0).

    Returns:
        lag_samples: lag in samples, fractional if subsample=True
            (positive = signal_b is delayed)
        lag_s: lag in seconds
        corr_value: Pearson correlation at the chosen lag
    """
    a_full = np.asarray(signal_a, dtype=np.float64)
    b_full = np.asarray(signal_b, dtype=np.float64)

    if search_radius_samples is None:
        search_radius_samples = max_lag_samples
    # convert the public "positive = b delayed" convention to the raw
    # internal convention _windowed_pearson_corr uses (see the note below).
    raw_center = int(round(-center_lag_samples))
    radius = int(round(search_radius_samples))
    lags = np.arange(raw_center - radius, raw_center + radius + 1)
    corrs = np.array([_windowed_pearson_corr(a_full, b_full, int(lag)) for lag in lags])
    best_idx = int(np.argmax(corrs))
    raw_lag = int(lags[best_idx])
    corr_value = float(corrs[best_idx])

    subsample_offset = 0.0
    if subsample and 0 < best_idx < len(corrs) - 1:
        y_minus, y_0, y_plus = corrs[best_idx - 1], corrs[best_idx], corrs[best_idx + 1]
        denom = (y_minus - 2 * y_0 + y_plus)
        if abs(denom) > 1e-12 and np.isfinite(denom):
            subsample_offset = 0.5 * (y_minus - y_plus) / denom

    # _windowed_pearson_corr(a, b, lag) compares a[lag:] against b[:n-lag],
    # i.e. tests "a advanced by lag samples matches b" - the empirical
    # opposite of this function's documented contract (verified: b[n]=a[n-15],
    # i.e. b truly delayed by +15 samples relative to a, produces raw_lag=-15
    # here). Negate to match "positive = signal_b is delayed".
    lag_samples = -(raw_lag + subsample_offset)
    lag_s = lag_samples / fs

    return lag_samples, lag_s, corr_value


def _dominant_frequency(x, fs, band_hz=(1.0, 25.0)):
    """Peak-power frequency of x within band_hz, via Welch PSD. Used to size
    the fine-stage search radius in find_time_offset_robust so it scales
    with the signal's own oscillation period instead of a hardcoded value."""
    freqs, pxx = signal.welch(np.asarray(x, dtype=np.float64), fs=fs,
                               nperseg=min(512, len(x)))
    band_mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(band_mask):
        return float(band_hz[0])
    band_freqs, band_pxx = freqs[band_mask], pxx[band_mask]
    return float(band_freqs[int(np.argmax(band_pxx))])


def _envelope(x, fs, band_hz, smooth_hz):
    """Hilbert-envelope of x band-limited to band_hz, then lowpass-smoothed
    to smooth_hz - the slowly-varying amplitude-modulation trace of a
    narrowband oscillation, used by find_time_offset_robust's coarse stage."""
    x_band = zero_phase_filter(x, fs=fs, cutoff_hz=band_hz, btype='band')
    env = np.abs(signal.hilbert(x_band))
    return zero_phase_filter(env, fs=fs, cutoff_hz=smooth_hz, btype='low')


def find_time_offset_robust(signal_a, signal_b, max_lag_samples, fs=50,
                             dominant_freq_hz=None, envelope_halfwidth_hz=4.0,
                             envelope_smooth_hz=1.5, refine_radius_periods=0.5):
    """
    Two-stage version of find_time_offset_by_xcorr, built to be stable
    against small upstream changes to a narrowband/periodic signal - a plain
    single-pass search over the full window is prone to landing on a
    periodicity-aliased peak nearly as tall as the true one (see that
    function's docstring), and *which* peak wins that comparison can flip
    from a barely-sub-sample shift in one of the inputs. That's not
    hypothetical here: fixing a phase bug in the LDS reference's own
    pre-processing (a genuine, confirmed, ~1-sample-scale group-delay
    correction) flipped the single-pass search onto a peak ~2.8 VIV periods
    away and regressed RMSE from 0.150mm to 0.209mm - see README
    "align_signals robustness".

    Stage 1 (coarse, alias-immune): rather than lowpassing to strip the
    oscillation and correlate whatever broadband content is left - tried
    first, and it doesn't work here: measured coherence between vision and
    LDS in this data is ~0.09 below 3Hz (independent per-channel drift, not
    shared signal) vs. ~0.89 at 10-13Hz (the VIV band itself is essentially
    the *only* place the two channels genuinely agree) - this instead
    extracts the Hilbert-envelope (amplitude-modulation trace) of the
    dominant oscillation itself, band-limited around dominant_freq_hz, from
    both signals. The envelope varies far slower than the oscillation (real
    amplitude buildup/decay, not noise), so a full-window search on it has
    one clear peak instead of a comb of period-spaced ones - and it's
    exploiting the one band that's actually coherent between vision and LDS.

    Stage 2 (fine, full precision): search a narrow window - at most half a
    VIV period on each side, so it structurally cannot reach a neighboring
    aliased peak - around the coarse estimate, on the original (unfiltered)
    signals, with the same sub-sample parabolic refinement as the plain
    search. This recovers full precision without reopening the ambiguity
    stage 1 was built to avoid.

    Args:
        signal_a, signal_b, max_lag_samples, fs: as in
            find_time_offset_by_xcorr (max_lag_samples bounds stage 1 only)
        dominant_freq_hz: the narrowband signal's dominant frequency, used
            both to center stage 1's envelope-extraction band and to size
            stage 2's search radius (one period = the spacing between
            aliased peaks). Auto-detected from signal_a via Welch PSD peak
            if not given.
        envelope_halfwidth_hz: half-width (Hz) of the band-pass used to
            isolate the oscillation before taking its envelope
        envelope_smooth_hz: lowpass cutoff (Hz) for smoothing the envelope
            itself, well below dominant_freq_hz
        refine_radius_periods: stage-2 half-width, in units of one period of
            dominant_freq_hz (0.5 = up to half a period either side of the
            coarse estimate - the widest radius that still can't reach a
            neighboring aliased peak)

    Returns:
        lag_samples, lag_s, corr_value: as in find_time_offset_by_xcorr, but
            from the constrained stage-2 search
        coarse_lag_samples: stage 1's own estimate, returned for diagnostics
    """
    a_full = np.asarray(signal_a, dtype=np.float64)
    b_full = np.asarray(signal_b, dtype=np.float64)

    if dominant_freq_hz is None:
        dominant_freq_hz = _dominant_frequency(a_full, fs)

    band_hz = (max(0.5, dominant_freq_hz - envelope_halfwidth_hz),
               min(fs / 2 * 0.98, dominant_freq_hz + envelope_halfwidth_hz))
    env_a = _envelope(a_full, fs, band_hz, envelope_smooth_hz)
    env_b = _envelope(b_full, fs, band_hz, envelope_smooth_hz)
    coarse_lag_samples, _, coarse_corr = find_time_offset_by_xcorr(
        env_a, env_b, max_lag_samples, fs=fs, subsample=False
    )

    period_samples = fs / dominant_freq_hz
    refine_radius_samples = max(1.0, refine_radius_periods * period_samples)

    lag_samples, lag_s, corr_value = find_time_offset_by_xcorr(
        a_full, b_full, max_lag_samples, fs=fs, subsample=True,
        center_lag_samples=coarse_lag_samples, search_radius_samples=refine_radius_samples
    )
    if os.environ.get('VIVBENCH_DEBUG_ALIGN'):
        print(f"[DEBUG align] dominant_freq_hz={dominant_freq_hz:.3f} period_samples={period_samples:.3f} "
              f"refine_radius_samples={refine_radius_samples:.3f} coarse_lag={coarse_lag_samples:.3f} "
              f"(coarse_corr={coarse_corr:.4f}) -> fine_lag={lag_samples:.3f} (corr={corr_value:.4f})")
    return lag_samples, lag_s, corr_value, coarse_lag_samples


def mean_subtract(signal_1d):
    """Remove DC offset."""
    return signal_1d - np.mean(signal_1d)


def project_2d_to_1d(displacement_2d, projection_axis_angle_rad):
    """
    Project 2D image displacement onto a 1D direction (e.g., perpendicular to cable).
    
    Args:
        displacement_2d: (N, 2) array where column 0 is x, column 1 is y
        projection_axis_angle_rad: angle in radians (0 = x-axis, π/2 = y-axis)
    
    Returns:
        displacement_1d: (N,) scalar projection
    """
    # Unit vector in projection direction
    axis = np.array([np.cos(projection_axis_angle_rad), np.sin(projection_axis_angle_rad)])
    
    # Project: dot product
    displacement_1d = np.dot(displacement_2d, axis)
    
    return displacement_1d


def interpolate_missing(data, max_gap=10):
    """
    Linear interpolation for NaN values.
    
    Args:
        data: 1D array with possible NaNs
        max_gap: max consecutive NaNs to interpolate (larger gaps left as NaN)
    
    Returns:
        interpolated: 1D array
    """
    idx = np.arange(len(data))
    valid = ~np.isnan(data)
    
    # Identify gaps
    interpolated = np.array(data, dtype=np.float64)
    i = 0
    while i < len(interpolated):
        if np.isnan(interpolated[i]):
            gap_start = i
            gap_end = i
            while gap_end < len(interpolated) and np.isnan(interpolated[gap_end]):
                gap_end += 1
            gap_size = gap_end - gap_start
            
            if gap_size <= max_gap and gap_start > 0 and gap_end < len(interpolated):
                # Linear interp
                left_val = interpolated[gap_start - 1]
                right_val = interpolated[gap_end]
                interpolated[gap_start:gap_end] = np.linspace(left_val, right_val, gap_size)
            
            i = gap_end
        else:
            i += 1
    
    return interpolated


def align_signals(displacement_mm, lds_mm, fs=50, max_lag_s=1.0, search_sign=True):
    """
    Find and trim overlap, apply mean-subtraction.

    Args:
        displacement_mm: vision-based displacement (mm)
        lds_mm: LDS reference (mm)
        fs: sampling rate (Hz)
        max_lag_s: max expected delay (seconds)
        search_sign: also try negating the vision signal and keep whichever
            polarity correlates better (see the note below). Leave True
            unless you have independently guaranteed the polarity.

    Returns:
        displacement_aligned: vision displacement (mean-subtracted, trimmed,
            and sign-corrected if search_sign found the flip fits better)
        lds_aligned: LDS reference (mean-subtracted, trimmed)
        lag_s: detected lag (seconds)
        valid_range: (start_idx, end_idx) in the overlapping window
        sign: +1.0, or -1.0 if the vision signal was negated to match LDS
    """
    max_lag_samples = int(max_lag_s * fs)

    # Ensure both have the same length (trim to shorter)
    min_len = min(len(displacement_mm), len(lds_mm))
    displacement_mm = displacement_mm[:min_len]
    lds_mm = lds_mm[:min_len]

    # Polarity search, then coarse-then-fine lag search per polarity.
    #
    # Searching the sign is NOT optional bookkeeping here, it is a
    # correctness requirement, and omitting it was a real and costly bug:
    # roi_io's viv_direction convention (normal forced to point "up") can be
    # the opposite polarity to the LDS's own sign convention, and for a
    # narrowband ~11.7Hz signal the correlation-vs-lag curve is itself
    # quasi-sinusoidal - so a polarity-flipped signal correlates almost as
    # well at a lag half a VIV period away (~43ms) as the true signal does
    # at the true lag. A sign-blind search silently locks onto that decoy.
    # Measured cost when it did: the aligner chose lag=+0.240s/sign=+1 over
    # the true lag=+0.200s/sign=-1, inflating wide-band (0.2-20Hz) RMSE from
    # 0.146mm to 0.265mm and 2-5Hz error from 0.028mm to 0.184mm - a
    # half-period shift is a modest phase error at 11.7Hz but a huge one at
    # 2-5Hz, which is why it disproportionately wrecked the low bands.
    #
    # find_time_offset_robust's envelope-based coarse stage cannot catch
    # this on its own: |hilbert(x)| == |hilbert(-x)|, so the coarse stage is
    # polarity-blind by construction, and its fine stage only searches +/-
    # half a period around that anchor - a window that contains the decoy.
    # The template-match + affine-flow configuration searches sign explicitly too.
    best = None
    for sgn in ((1.0, -1.0) if search_sign else (1.0,)):
        cand_lag_samples, cand_lag_s, cand_corr, _ = find_time_offset_robust(
            lds_mm, sgn * displacement_mm, max_lag_samples, fs=fs
        )
        if best is None or cand_corr > best[2]:
            best = (cand_lag_samples, cand_lag_s, cand_corr, sgn)

    lag_samples, lag_s, _, sign = best
    displacement_mm = sign * displacement_mm

    # Sub-sample shift: vision lags lds by lag_samples, so shift vision
    # earlier in time by lag_samples to bring it into alignment. Cubic-spline
    # interpolation handles the fractional part; 'nearest' padding avoids
    # ringing at the edges (which get trimmed off below anyway).
    displacement_shifted = ndimage.shift(displacement_mm, shift=-lag_samples, order=3, mode='nearest')

    # Drop the edge region that was extrapolated by the shift, plus a couple
    # of samples of margin.
    edge_margin = int(np.ceil(abs(lag_samples))) + 2
    if lag_samples >= 0:
        displacement_aligned = displacement_shifted[edge_margin:]
        lds_aligned = lds_mm[edge_margin:]
    else:
        displacement_aligned = displacement_shifted[:-edge_margin] if edge_margin > 0 else displacement_shifted
        lds_aligned = lds_mm[:-edge_margin] if edge_margin > 0 else lds_mm

    # Ensure same length
    min_len = min(len(displacement_aligned), len(lds_aligned))
    displacement_aligned = displacement_aligned[:min_len]
    lds_aligned = lds_aligned[:min_len]
    
    # Mean-subtract
    displacement_aligned = mean_subtract(displacement_aligned)
    lds_aligned = mean_subtract(lds_aligned)
    
    valid_range = (0, min_len)

    return displacement_aligned, lds_aligned, lag_s, valid_range, sign


def align_signals_native_lds(displacement_mm, lds_decimated, lds_native, fs_lds_native,
                              fs=50, max_lag_s=1.0, polish_radius_s=0.05, polish_step_s=None):
    """
    Like align_signals, but for the fine, precision-critical part of the lag
    search, the LDS reference comes from direct interpolation of its native
    (undecimated) samples instead of the pre-decimated 50Hz array - this is
    what the template-match + affine-flow configuration does (it never decimates LDS at all;
    see README "direct LDS interpolation"), tested here to check whether
    our own decimation - even zero-phase-corrected (see decimate_lds) - is
    itself limiting precision, independent of the alignment-search
    correctness already fixed separately (find_time_offset_robust).

    Two stages:
      1. Coarse, alias-robust anchor: find_time_offset_robust on the
         pre-decimated pair, exactly as align_signals does - decimation's
         own small phase bias doesn't matter at this resolution.
      2. Fine polish: scan lag_s on a dense grid within polish_radius_s of
         that anchor; at each candidate, interpolate lds_native directly at
         (t_vis - lag_s) (see module-level note below on why minus, not
         plus) and score by the same windowed Pearson correlation - so the
         LDS values used for the final answer never pass through the
         decimation filter at all, only interpolation of the original
         10kHz-sampled data (linear interpolation of a signal this
         oversampled relative to its own ~11.7Hz content is essentially
         exact - not a meaningful new distortion source).

    Args:
        displacement_mm: vision displacement (mm), 1D, uniform at `fs` Hz starting at t=0
        lds_decimated: LDS pre-decimated to `fs` Hz (coarse stage only)
        lds_native: LDS at its native (undecimated) sample rate
        fs_lds_native: native LDS sample rate (Hz)
        fs: vision sample rate (Hz)
        max_lag_s: coarse-stage search half-width (seconds)
        polish_radius_s: fine-stage search half-width around the coarse
            anchor (seconds) - a handful of vision samples, not a full VIV
            period (that robustness already comes from stage 1)
        polish_step_s: fine-stage grid step (seconds); default 1/(fs*20)
            (20x the vision sample rate)

    Returns:
        displacement_aligned, lds_aligned, lag_s, valid_range - same
        contract as align_signals (lds_aligned here is native-interpolated,
        not a decimated-array slice)
    """
    max_lag_samples = int(max_lag_s * fs)
    min_len = min(len(displacement_mm), len(lds_decimated))
    displacement_mm = displacement_mm[:min_len]
    lds_decimated = lds_decimated[:min_len]

    coarse_lag_samples, coarse_lag_s, _, _ = find_time_offset_robust(
        lds_decimated, displacement_mm, max_lag_samples, fs=fs
    )

    t_vis = np.arange(len(displacement_mm)) / fs
    t_lds_native = np.arange(len(lds_native)) / fs_lds_native
    lds_native = np.asarray(lds_native, dtype=np.float64)
    vision0 = displacement_mm - np.mean(displacement_mm)

    if polish_step_s is None:
        polish_step_s = 1.0 / (fs * 20)
    candidate_lags = np.arange(coarse_lag_s - polish_radius_s,
                                coarse_lag_s + polish_radius_s + polish_step_s, polish_step_s)

    # See align_signals: aligned_vision(t) = vision(t + lag_s) paired against
    # aligned_lds(t) = lds(t) there. Equivalently (substituting s = t + lag_s):
    # vision(s) pairs against lds(s - lag_s) - so here, where vision samples
    # are used unshifted, LDS must be evaluated at (t_vis - lag_s) to match
    # the same convention. Verified to reproduce align_signals' own lag_s
    # and aligned arrays on synthetic data before trusting on real data.
    # Scored on |corr|, keeping the sign, so the polarity search is free here
    # (a sign flip just negates corr) - see align_signals' long note on why
    # searching polarity is a correctness requirement, not bookkeeping.
    best_corr, best_lag_s, best_sign = -np.inf, coarse_lag_s, 1.0
    for lag_s in candidate_lags:
        tt = t_vis - lag_s
        mask = (tt >= t_lds_native[0]) & (tt <= t_lds_native[-1])
        if mask.sum() < 20:
            continue
        lds_i = np.interp(tt[mask], t_lds_native, lds_native)
        v = vision0[mask]
        v0 = v - v.mean()
        l0 = lds_i - lds_i.mean()
        denom = np.sqrt(np.sum(v0 ** 2) * np.sum(l0 ** 2))
        if denom < 1e-12:
            continue
        corr = float(np.dot(v0, l0) / denom)
        if abs(corr) > best_corr:
            best_corr, best_lag_s = abs(corr), float(lag_s)
            best_sign = 1.0 if corr >= 0 else -1.0

    lag_s, sign = best_lag_s, best_sign
    tt = t_vis - lag_s
    mask = (tt >= t_lds_native[0]) & (tt <= t_lds_native[-1])
    lds_aligned = np.interp(tt[mask], t_lds_native, lds_native)
    displacement_aligned = sign * displacement_mm[mask]

    min_len = min(len(displacement_aligned), len(lds_aligned))
    displacement_aligned = mean_subtract(displacement_aligned[:min_len])
    lds_aligned = mean_subtract(lds_aligned[:min_len])
    valid_range = (0, min_len)

    return displacement_aligned, lds_aligned, lag_s, valid_range, sign
