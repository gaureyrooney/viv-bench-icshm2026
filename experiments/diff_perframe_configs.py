"""
Per-frame diff: our best signal vs. the template-match + affine-flow configuration's reproduced signal, both against
LDS ground truth - to find exactly where/when they diverge, ratits than
inferring it indirectly from aggregate RMSE.

Reuses its already-saved tracking_full.csv (from run_templatematch_affine_config.py) and
its own compare_with_lds/plot_results alignment logic (copied verbatim, not a
reimplementation) to reconstruct the exact aligned arrays that produced her
0.07039mm result - then cross-correlates our own aligned signal against the template-match + affine-flow configuration's
to find their relative time offset (since the two pipelines trim/align to
LDS slightly differently) and overlays both.

Does NOT modify run_templatematch_affine_config.py.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import video_path, lds_path, templatematch_affine_script, templatematch_affine_config_json
from dsp.alignment import find_time_offset_by_xcorr, zero_phase_filter

TM_AFFINE_SCRIPT = templatematch_affine_script()
TM_AFFINE_TRACKING_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'results', 'templatematch_affine_config', 'tracking_full.csv')
LDS_PATH = lds_path()
OUR_RUN = 'icgn_differential_bp9-14_symlds'  # icgn+differential tracking/alignment is deterministic;
# this is just the first icgn+differential run with aligned_signals.npz saved (post Stage-2). Its
# prefilter/aligned arrays are identical to the genuine (non-symlds) icgn_differential leaderboard
# entry (0.14994mm) - symmetric_lds_filter only changes which final metric gets reported, not the
# saved prefilter tracking/alignment arrays.


def load_config_module(path):
    spec = importlib.util.spec_from_file_location("tm_affine_config_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reconstruct_tm_affine_aligned(sb, highpass_hz=0.20, lowpass_hz=20.0):
    """Mirrors its main()/plot_results() alignment logic exactly (same few
    lines, not reimplemented) to recover the (time, vision, lds) arrays that
    produced its reported RMSE - not just the scalar summary."""
    df = pd.read_csv(TM_AFFINE_TRACKING_CSV)
    t = df['time_s'].to_numpy(np.float64)
    y0 = df['displacement_mm_motion_compensated'].to_numpy(np.float64)
    y0 = y0 - np.nanmean(y0)
    y_proc = sb.filter_signal(y0, fs=50.0, highpass_hz=highpass_hz, lowpass_hz=lowpass_hz)
    submission_y = y_proc - np.nanmean(y_proc)

    t_lds, y_lds = sb.read_lds_xlsx(LDS_PATH, lds_fs=10000.0)
    y_lds_proc = sb.filter_signal(y_lds, fs=10000.0, highpass_hz=highpass_hz, lowpass_hz=lowpass_hz)
    best = sb.compare_with_lds(t, submission_y, t_lds, y_lds_proc, fps=50.0, max_lag_s=1.0)

    # --- exact logic from its plot_results(), reconstructing the aligned arrays ---
    lag, sign = best['lag_s'], best['sign']
    tt = t + lag
    mask = (tt >= t_lds[0]) & (tt <= t_lds[-1])
    yv = sign * (submission_y - np.nanmean(submission_y))
    yl = y_lds_proc - np.nanmean(y_lds_proc)
    yl_i = np.interp(tt[mask], t_lds, yl)
    vv = yv[mask] - np.mean(yv[mask])
    yl_i = yl_i - np.mean(yl_i)

    return {'time_lds_clock': tt[mask], 'vision': vv, 'lds': yl_i, 'best': best}


def main():
    sb = load_config_module(TM_AFFINE_SCRIPT)
    its = reconstruct_tm_affine_aligned(sb)
    print(f"templatematch_affine_config reconstruction check: rmse={her['best']['rmse_mm']:.5f} "
          f"corr={her['best']['corr']:.5f} (want 0.07039 / 0.98290)")

    ours_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', OUR_RUN, 'aligned_signals.npz')
    ours = np.load(ours_path)
    our_disp_raw = ours['displacement_mm']
    our_lds_raw = ours['lds_reference']

    her_lds_bp = zero_phase_filter(her['lds'], fs=50, cutoff_hz=(9.0, 14.0), btype='band')
    our_lds_bp = zero_phase_filter(our_lds_raw, fs=50, cutoff_hz=(9.0, 14.0), btype='band')
    # find_time_offset_by_xcorr indexes both arrays as if same length; its LDS
    # (masked to compare_with_lds's overlap, ~3000 samples) and ours (trimmed
    # by our own align_signals, independently sized) generally differ - clip
    # to the shorter one just for finding the offset (full arrays used after).
    n_probe = min(len(her_lds_bp), len(our_lds_bp))
    lag_samples, lag_s, corr_peak = find_time_offset_by_xcorr(
        her_lds_bp[:n_probe], our_lds_bp[:n_probe], max_lag_samples=50, fs=50
    )
    print(f"LDS-to-LDS cross-check offset between the two pipelines' own LDS arrays: "
          f"{lag_samples:.2f} samples ({lag_s*1000:.1f} ms), peak corr={corr_peak:.4f}")

    n = len(her['vision'])
    shift = int(round(lag_samples))
    if shift >= 0:
        her_v = her['vision'][shift:]
        her_l = her['lds'][shift:]
        our_v = our_disp_raw[:len(her_v)]
        our_l = our_lds_raw[:len(her_v)]
    else:
        her_v = her['vision'][:shift]
        her_l = her['lds'][:shift]
        our_v = our_disp_raw[-shift:-shift + len(her_v)]
        our_l = our_lds_raw[-shift:-shift + len(her_v)]

    n = min(len(her_v), len(our_v))
    her_v, her_l, our_v, our_l = her_v[:n], her_l[:n], our_v[:n], our_l[:n]
    print(f"Common overlap window after alignment: {n} samples ({n/50:.1f}s)")

    her_v_bp = zero_phase_filter(her_v, fs=50, cutoff_hz=(9.0, 14.0), btype='band')
    our_v_bp = zero_phase_filter(our_v, fs=50, cutoff_hz=(9.0, 14.0), btype='band')
    lds_common = zero_phase_filter(her_l, fs=50, cutoff_hz=(9.0, 14.0), btype='band')

    residual = our_v_bp - her_v_bp
    print(f"\nOur signal RMSE vs LDS (this window): {np.sqrt(np.mean((our_v_bp-lds_common)**2)):.4f} mm")
    print(f"Her signal RMSE vs LDS (this window):  {np.sqrt(np.mean((her_v_bp-lds_common)**2)):.4f} mm")
    print(f"Our-vs-its direct RMSE:                {np.sqrt(np.mean(residual**2)):.4f} mm")
    print(f"Our-vs-its correlation:                {np.corrcoef(our_v_bp, her_v_bp)[0,1]:.4f}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'diff_perframe_configs')
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, 'diff_arrays.npz'),
             time_s=np.arange(n)/50.0, our_v_bp=our_v_bp, her_v_bp=her_v_bp,
             lds_common=lds_common, residual=residual)
    print(f"\nSaved to {out_dir}/diff_arrays.npz")


if __name__ == '__main__':
    main()
