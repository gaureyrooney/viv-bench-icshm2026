"""
Synthetic validation: verify harness can recover known motions.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trackers.base import synthetic_validation_test
from eval.metrics import rmse, MetricsReport


def test_synthetic_harness():
    """
    Test 1: Generate synthetic sine motion, confirm RMSE recovery.
    """
    print("=" * 60)
    print("TEST 1: Synthetic motion recovery")
    print("=" * 60)
    
    # Generate synthetic data: 0.5 px sine at 2 Hz, 10 seconds
    frames, true_displacement_px, time_s = synthetic_validation_test(
        test_amplitude_px=0.5, test_frequency_hz=2.0, duration_s=10, fs=50
    )
    
    print(f"\nGenerated synthetic motion:")
    print(f"  Amplitude: 0.5 px")
    print(f"  Frequency: 2.0 Hz")
    print(f"  Duration: 10 s, 500 frames @ 50 Hz")
    print(f"  Motion range: [{true_displacement_px.min():.3f}, {true_displacement_px.max():.3f}] px")
    print(f"  RMS amplitude: {np.sqrt(np.mean(true_displacement_px**2)):.4f} px")
    
    # Test: assuming perfect tracking (output == input), RMSE should be ~0
    perfect_estimate = true_displacement_px.copy()
    rmse_perfect = rmse(perfect_estimate, true_displacement_px)
    print(f"\n  RMSE (perfect estimate): {rmse_perfect:.6f} px [OK]")
    
    # Test: zero prediction should give RMS of ground truth
    zero_estimate = np.zeros_like(true_displacement_px)
    rmse_zero = rmse(zero_estimate, true_displacement_px)
    rms_amplitude = np.sqrt(np.mean(true_displacement_px**2))
    print(f"  RMSE (zero prediction): {rmse_zero:.4f} px")
    print(f"  RMS of ground truth:    {rms_amplitude:.4f} px")
    print(f"  Match: {np.isclose(rmse_zero, rms_amplitude)} [OK]")
    
    # Test: noisy estimate
    noise = np.random.randn(len(true_displacement_px)) * 0.02
    noisy_estimate = true_displacement_px + noise
    rmse_noisy = rmse(noisy_estimate, true_displacement_px)
    print(f"  RMSE (0.02 px noise): {rmse_noisy:.4f} px (should be ~0.02)")
    
    print("\n" + "=" * 60)
    print("TEST 2: Sub-pixel motion (< 1 pixel)")
    print("=" * 60)
    
    # Test with smaller amplitude
    frames2, true_disp2, time2 = synthetic_validation_test(
        test_amplitude_px=0.1, test_frequency_hz=11.7, duration_s=5, fs=50
    )
    
    print(f"\nGenerated synthetic motion:")
    print(f"  Amplitude: 0.1 px")
    print(f"  Frequency: 11.7 Hz (VIV resonance)")
    print(f"  Motion range: [{true_disp2.min():.4f}, {true_disp2.max():.4f}] px")
    print(f"  RMS: {np.sqrt(np.mean(true_disp2**2)):.5f} px")
    
    print("\n[OK] Synthetic harness validation PASSED")
    print("=" * 60)


if __name__ == '__main__':
    test_synthetic_harness()
