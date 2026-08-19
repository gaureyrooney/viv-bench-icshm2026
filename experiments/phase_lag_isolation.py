"""
Isolate the source of the ~4deg/~1ms phase lag at 11.7Hz found by
diff_perframe_configs.py between our icgn+differential signal and the template-match + affine-flow configuration's
reproduced signal.

Compares 11.7Hz phase (relative to LDS) across the 2x2 tracker x compensator
grid (icgn/template_match x differential/affine_flow), using each run's
saved aligned_signals.npz (pre-filter, own-alignment) so every combo is
measured the same way as the original diff. Whichever swap (tracker vs
compensator) moves the phase back toward LDS's phase identifies the source.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsp.alignment import zero_phase_filter

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

RUNS = {
    'icgn + differential':        'icgn_differential_bp9-14_symlds',
    'template_match + differential': 'templatematch_differential_bp9-14_symlds',
    'icgn + affine_flow':         'icgn_affineflow_bp9-14_symlds',
    'template_match + affine_flow': 'templatematch_affineflow_bp9-14_phaseprobe',
}


def amp_phase_at(x, f0, fs):
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    X = np.fft.rfft(x * np.hanning(n))
    idx = int(np.argmin(np.abs(freqs - f0)))
    return float(np.abs(X[idx])), float(np.angle(X[idx], deg=True)), float(freqs[idx])


def main():
    fs = 50.0
    print(f"{'combo':<32} {'amp':>8} {'phase(deg)':>11} {'dphi vs LDS':>12} {'rmse_vs_lds':>12}")
    lds_phase_ref = None
    for label, run_name in RUNS.items():
        npz_path = os.path.join(RESULTS_DIR, run_name, 'aligned_signals.npz')
        if not os.path.exists(npz_path):
            print(f"{label:<32} MISSING ({run_name})")
            continue
        d = np.load(npz_path)
        disp, lds = d['displacement_mm'], d['lds_reference']

        disp_bp = zero_phase_filter(disp, fs=fs, cutoff_hz=(9.0, 14.0), btype='band')
        lds_bp = zero_phase_filter(lds, fs=fs, cutoff_hz=(9.0, 14.0), btype='band')

        amp_v, ph_v, f_actual = amp_phase_at(disp_bp, 11.7, fs)
        amp_l, ph_l, _ = amp_phase_at(lds_bp, 11.7, fs)
        if lds_phase_ref is None:
            lds_phase_ref = ph_l  # each run's own LDS slice - report per-run for sanity

        dphi = ((ph_v - ph_l + 180) % 360) - 180
        rmse = float(np.sqrt(np.mean((disp_bp - lds_bp) ** 2)))
        print(f"{label:<32} {amp_v:8.2f} {ph_v:11.2f} {dphi:12.2f} {rmse:12.4f}   (own LDS phase={ph_l:.2f}deg)")


if __name__ == '__main__':
    main()
