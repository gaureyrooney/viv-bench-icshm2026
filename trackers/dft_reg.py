"""
DFT-based phase correlation tracker (Tier-1).

Uses upsampled phase correlation for sub-pixel accuracy.
Reference: Guizar-Sicairos et al., "Efficient subpixel image registration algorithms"
"""
import numpy as np
from scipy import ndimage, signal
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trackers.base import Tracker


def phase_correlation_upsample(im1, im2, upsample_factor=100, peak_margin=6, max_shift=10):
    """
    Sub-pixel phase correlation using Fourier upsampling.

    Args:
        im1: reference frame (grayscale uint8)
        im2: current frame (grayscale uint8)
        upsample_factor: refinement factor (higher = finer sub-pixel accuracy)
        peak_margin: half-width (in original pixels) of the crop around the
            coarse peak that gets Fourier-upsampled. This is independent of
            upsample_factor - conflating the two used to force e.g. a
            6400x6400-point FFT per call when upsample_factor=100, which is
            intractable over thousands of frames. A small margin (a few
            pixels) is enough since true motion is sub-pixel to begin with.
        max_shift: constrain the *coarse* peak search to displacements within
            +/- max_shift px of zero. Repeating patterns (like a checkerboard
            target) create multiple equally-strong correlation peaks at the
            pattern's spatial period; an unconstrained global argmax can lock
            onto one of those instead of the true near-zero peak, producing a
            large spurious "jump" every call. Since displacements here are
            known a priori to be small (sub-pixel VIV, or a UAV-jitter frame
            step), restricting the search window is physically justified, not
            just a hack. Set to None to disable (unconstrained global search).

    Returns:
        (dx, dy): sub-pixel displacement
        error: normalized cross-correlation peak
    """
    # NOTE on the upsampling step below: np.fft.ifft2(np.fft.fft2(x, s=S)) is
    # a mathematical no-op when both calls use the same target shape S - it
    # just returns x zero-padded to S (verified: fft2 followed by ifft2 of the
    # same transform pair is the identity). An earlier version of this
    # function relied on exactly that pattern and called it "Fourier zoom",
    # which never actually interpolated anything (confirmed via a
    # self-correlation sanity check: correlating an image against itself
    # returned a nonzero "peak", when it must be exactly zero). Real Fourier
    # upsampling requires zero-padding the *spectrum* (after fftshift), not
    # the spatial array before the forward FFT - that's what happens below.

    # Convert to float and apply a Hann window to reduce edge/periodicity
    # leakage in the FFT (helps with the checkerboard's repeating texture).
    f1 = im1.astype(np.float64)
    f2 = im2.astype(np.float64)
    win_y = np.hanning(f1.shape[0])
    win_x = np.hanning(f1.shape[1])
    window = np.outer(win_y, win_x)
    f1w = f1 * window
    f2w = f2 * window

    # FFTs
    F1 = np.fft.fft2(f1w)
    F2 = np.fft.fft2(f2w)

    # Cross-power spectrum (normalized)
    cross_power = F1 * np.conj(F2)
    cross_power_norm = cross_power / (np.abs(cross_power) + 1e-10)

    # Inverse FFT to get phase correlation
    phase_corr = np.fft.ifft2(cross_power_norm).real
    h, w = phase_corr.shape

    # Find coarse peak, optionally restricted to a small capture window
    # around zero-shift (indices near 0 and near h/w, i.e. small +/- shifts).
    if max_shift is not None:
        search_space = np.full_like(phase_corr, -np.inf)
        ys = np.r_[0:min(max_shift + 1, h), max(0, h - max_shift):h]
        xs = np.r_[0:min(max_shift + 1, w), max(0, w - max_shift):w]
        search_space[np.ix_(ys, xs)] = phase_corr[np.ix_(ys, xs)]
    else:
        search_space = phase_corr
    peak_coarse = np.unravel_index(np.argmax(search_space), search_space.shape)

    # Center the coarse peak via roll (correctly handles FFT wraparound,
    # unlike clamped slicing) then crop a small window for upsampling.
    margin = peak_margin
    crop_size = 2 * margin + 1
    centered = np.roll(np.roll(phase_corr, margin - peak_coarse[0], axis=0),
                        margin - peak_coarse[1], axis=1)
    phase_crop = centered[:crop_size, :crop_size]

    # Real Fourier-domain upsampling: pad the *spectrum*, not the spatial
    # array, so the inverse FFT sinc-interpolates the crop onto a finer grid.
    crop_h, crop_w = phase_crop.shape
    new_h, new_w = crop_h * upsample_factor, crop_w * upsample_factor
    spectrum = np.fft.fftshift(np.fft.fft2(phase_crop))
    spectrum_padded = np.zeros((new_h, new_w), dtype=complex)
    y0 = (new_h - crop_h) // 2
    x0 = (new_w - crop_w) // 2
    spectrum_padded[y0:y0 + crop_h, x0:x0 + crop_w] = spectrum
    phase_upsampled = np.abs(
        np.fft.ifft2(np.fft.ifftshift(spectrum_padded)) * (upsample_factor ** 2)
    )

    # Find peak in upsampled image, convert back to original-pixel offset
    # relative to the coarse peak (position `margin` in the crop == peak_coarse).
    peak_upsampled = np.unravel_index(np.argmax(phase_upsampled), phase_upsampled.shape)
    dy = (peak_upsampled[0] / upsample_factor - margin) + peak_coarse[0]
    dx = (peak_upsampled[1] / upsample_factor - margin) + peak_coarse[1]

    # Handle wrapping: convert to signed displacements
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w

    # Sign convention: cross_power = F1 * conj(F2) puts the correlation peak
    # at the shift that aligns im2 back onto im1, which is the *negative* of
    # how far im2's content has moved relative to im1. Callers want "how much
    # did the target move from im1 to im2", so flip the sign here once
    # (verified against a known synthetic shift: recovering a content shift
    # of (dy=-2, dx=3) gave (dx=-3, dy=2) before this negation).
    dx, dy = -dx, -dy

    # Confidence: how sharply peaked the correlation surface is (peak vs. the
    # typical background level). The previous formula divided the peak by
    # norm(f1)*norm(f2) - two ~64x64 pixel-intensity norms - which are many
    # orders of magnitude larger than any phase-correlation value, so it
    # always evaluated to ~1e-8 regardless of match quality.
    error = float(np.clip(phase_corr.max() / (np.mean(np.abs(phase_corr)) + 1e-10) / 20.0, 0.0, 1.0))

    return dx, dy, error


class DFTRegistrationTracker(Tracker):
    """
    Phase correlation based on Guizar-Sicairos upsampled FFT.
    
    Tracks by measuring pixel displacement between consecutive frames
    and integrating over time.
    """
    
    def __init__(self, name='DFT-Registration', upsample_factor=100, roi_size=64, max_shift=10):
        super().__init__(name, max_frames=None)
        self.upsample_factor = upsample_factor
        self.roi_size = roi_size
        self.max_shift = max_shift
    
    def track(self, frames, query_points):
        """
        Track query points by registering ROIs around each point.
        
        Args:
            frames: (T, H, W) grayscale frames
            query_points: (N, 2) query coordinates (x, y)
        
        Returns:
            trajectories: (T, N, 2) tracked positions
            confidence: (T, N) confidence scores
            meta: dict with metadata
        """
        T, H, W = frames.shape
        N = len(query_points)
        
        trajectories = np.zeros((T, N, 2), dtype=np.float32)
        confidence = np.zeros((T, N), dtype=np.float32)
        
        # ROI size (must be square and power of 2 for FFT efficiency)
        roi_size = self.roi_size
        
        # Initialize trajectories at query points
        trajectories[0, :, :] = query_points
        confidence[0, :] = 1.0
        
        # Reference frame (frame 0)
        frame_ref = frames[0]
        
        # For each subsequent frame
        for t in range(1, T):
            frame_current = frames[t]
            
            for n in range(N):
                # Get reference ROI
                x_ref, y_ref = query_points[n]
                x_ref_int, y_ref_int = int(np.round(x_ref)), int(np.round(y_ref))
                
                # Extract ROI from reference and current frame
                roi_ref_y = slice(max(0, y_ref_int - roi_size // 2), 
                                  min(H, y_ref_int + roi_size // 2))
                roi_ref_x = slice(max(0, x_ref_int - roi_size // 2), 
                                  min(W, x_ref_int + roi_size // 2))
                
                roi_ref = frame_ref[roi_ref_y, roi_ref_x]
                
                # Get current position (from previous frame)
                x_curr, y_curr = trajectories[t-1, n, :]
                x_curr_int, y_curr_int = int(np.round(x_curr)), int(np.round(y_curr))
                
                roi_curr_y = slice(max(0, y_curr_int - roi_size // 2), 
                                   min(H, y_curr_int + roi_size // 2))
                roi_curr_x = slice(max(0, x_curr_int - roi_size // 2), 
                                   min(W, x_curr_int + roi_size // 2))
                
                roi_curr = frame_current[roi_curr_y, roi_curr_x]
                
                # Skip if ROI is too small
                if roi_ref.shape[0] < 16 or roi_ref.shape[1] < 16 or \
                   roi_curr.shape[0] < 16 or roi_curr.shape[1] < 16:
                    trajectories[t, n, :] = trajectories[t-1, n, :]
                    confidence[t, n] = 0.1
                    continue
                
                # Pad to same size if needed
                if roi_ref.shape != roi_curr.shape:
                    size_y = max(roi_ref.shape[0], roi_curr.shape[0])
                    size_x = max(roi_ref.shape[1], roi_curr.shape[1])
                    roi_ref_padded = np.zeros((size_y, size_x), dtype=roi_ref.dtype)
                    roi_curr_padded = np.zeros((size_y, size_x), dtype=roi_curr.dtype)
                    roi_ref_padded[:roi_ref.shape[0], :roi_ref.shape[1]] = roi_ref
                    roi_curr_padded[:roi_curr.shape[0], :roi_curr.shape[1]] = roi_curr
                    roi_ref, roi_curr = roi_ref_padded, roi_curr_padded
                
                # Compute phase correlation
                try:
                    dx, dy, error = phase_correlation_upsample(
                        roi_ref, roi_curr, upsample_factor=self.upsample_factor,
                        max_shift=self.max_shift
                    )
                    
                    # Update position
                    trajectories[t, n, 0] = trajectories[t-1, n, 0] + dx
                    trajectories[t, n, 1] = trajectories[t-1, n, 1] + dy
                    confidence[t, n] = min(1.0, error)
                    
                except Exception as e:
                    # Fallback: use previous position
                    trajectories[t, n, :] = trajectories[t-1, n, :]
                    confidence[t, n] = 0.0
        
        meta = {
            'tracker': self.name,
            'roi_size': roi_size,
            'upsample_factor': self.upsample_factor,
        }
        
        return trajectories, confidence, meta
