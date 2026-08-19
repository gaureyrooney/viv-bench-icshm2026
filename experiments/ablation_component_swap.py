"""
2x2 ablation: whose TARGET TRACKING and whose CAMERA COMPENSATION accounts for
the remaining gap to the template-match + affine-flow configuration's 0.070mm?

Established by earlier diagnostics (see README "metric convention" section):
  - our alignment + LDS handling are NOT the problem (feeding her raw vision
    signal through our align_signals + our LDS reproduces 0.0717mm)
  - our error is concentrated below 5Hz, and is *identical* between our two
    compensators at 2-5/5-8/9-14Hz - implicating target tracking, not
    compensation, for those bands

This isolates it directly by swapping the two components across pipelines:
    her target + her affine   (= her published result, control)
    her target + our differential
    our target + her affine
    our target + our differential   (= our result, control)

Runs her code unmodified via importlib (same trick as run_templatematch_affine_config.py,
which is left untouched - it is the exact-reproduction regression asset), one
video pass, dumping her per-frame target centre and affine matrix, which her
CSV export in run_templatematch_affine_config.py doesn't retain.
"""
import importlib.util
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import video_path, lds_path, templatematch_affine_script, templatematch_affine_config_json
from data.lds import load_lds_prepared
from dsp.alignment import align_signals, zero_phase_filter

TM_AFFINE_SCRIPT = templatematch_affine_script()
VIDEO = video_path()
LDS = lds_path()
OUR_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'results', 'icgn_differential_bp9-14_v2', 'trajectories.npz')
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'ablation_component_swap', 'her_perframe.npz')

# from her saved templatematch_affine_config_json.json (same values run_templatematch_affine_config.py uses)
TARGET_ROI = (1867, 857, 65, 62)
STAB_ROIS = [(1354, 811, 67, 67), (2472, 1303, 67, 70)]
MM_PER_PX = 0.8661417322834646
SEARCH_MARGIN = 70
U_NORMAL = np.array([-0.45016305, 0.89294638])
OUR_TARGET_ORIGIN = np.array([1850.0, 838.0])   # data/video.py: round(centre - patch_size/2)
BANDS = [(0.2, 1), (1, 2), (2, 5), (5, 8), (9, 14), (14, 20)]


def load_config_module(path):
    spec = importlib.util.spec_from_file_location("tm_affine_config_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def collect_her_perframe(sb):
    """One video pass with her unmodified functions, keeping the per-frame
    target centre and affine matrix (her CSV export drops both)."""
    cap = cv2.VideoCapture(VIDEO)
    ret, first = cap.read()
    if not ret:
        raise RuntimeError("could not read first frame")
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    target_state = sb.make_target_state(first, TARGET_ROI)   # takes BGR, not gray
    pts0 = sb.build_static_feature_points(first_gray, STAB_ROIS)
    print(f"    static feature points: {len(pts0)}")

    centres, mats = [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        c_img, _, _ = sb.track_target_template(gray, target_state, SEARCH_MARGIN)
        M, _, _ = sb.estimate_camera_affine(first_gray, gray, pts0)
        centres.append(c_img)
        mats.append(M)
        if idx % 500 == 0:
            print(f"    frame {idx}")
        idx += 1
    cap.release()
    return np.array(centres), np.array(mats)


def apply_affine_seq(centres, mats):
    """Her compensation: stabilised = inverse_affine(M) applied to the centre."""
    out = np.zeros_like(centres)
    for t in range(len(centres)):
        Minv = cv2.invertAffineTransform(mats[t])
        x, y = centres[t]
        out[t] = [Minv[0, 0] * x + Minv[0, 1] * y + Minv[0, 2],
                  Minv[1, 0] * x + Minv[1, 1] * y + Minv[1, 2]]
    return out


def apply_differential(centres, bg_traj):
    """Our compensation: subtract the mean motion of the background points."""
    bg_mean = bg_traj.mean(axis=0)
    return centres - (bg_mean - bg_mean[0])


def score(label, stabilised, lds):
    disp = (stabilised - stabilised[0]) @ U_NORMAL * MM_PER_PX
    v, l, lag, _, _ = align_signals(disp, lds, fs=50, max_lag_s=1.0)  # now searches polarity itself
    per_band = []
    for lo, hi in BANDS:
        vb = zero_phase_filter(v, fs=50, cutoff_hz=(lo, hi), btype='band')
        lb = zero_phase_filter(l, fs=50, cutoff_hz=(lo, hi), btype='band')
        per_band.append(f"{lo}-{hi}:{np.sqrt(np.mean((vb - lb) ** 2)):.4f}")
    vw = zero_phase_filter(v, fs=50, cutoff_hz=(0.2, 20.), btype='band')
    lw = zero_phase_filter(l, fs=50, cutoff_hz=(0.2, 20.), btype='band')
    wide = np.sqrt(np.mean((vw - lw) ** 2))
    print(f"{label:34s} " + " ".join(per_band) + f"  || sym0.2-20 = {wide:.4f} mm")
    return wide


def main():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        her_centres, her_mats = z['centres'], z['mats']
        print(f"[cache] reusing her per-frame data ({len(her_centres)} frames)")
    else:
        sb = load_config_module(TM_AFFINE_SCRIPT)
        print("[run] one video pass with her unmodified tracking functions...")
        her_centres, her_mats = collect_her_perframe(sb)
        np.savez(CACHE, centres=her_centres, mats=her_mats)

    ours = np.load(OUR_CACHE)
    our_centres = ours['target_trajectory_px'].reshape(-1, 2) + OUR_TARGET_ORIGIN
    bg_traj = np.stack([b.reshape(-1, 2) for b in ours['bg_trajectories_px']])

    n = min(len(her_centres), len(our_centres), bg_traj.shape[1])
    her_centres, her_mats = her_centres[:n], her_mats[:n]
    our_centres, bg_traj = our_centres[:n], bg_traj[:, :n]

    lds, _, _ = load_lds_prepared(LDS)
    print(f"\n{n} common frames\n")
    print(f"{'variant':34s} " + " ".join(f"{lo}-{hi}" for lo, hi in BANDS))
    score("HER target + HER affine  (ctrl)", apply_affine_seq(her_centres, her_mats), lds)
    score("HER target + OUR differential", apply_differential(her_centres, bg_traj), lds)
    score("OUR target + HER affine", apply_affine_seq(our_centres, her_mats), lds)
    score("OUR target + OUR differential", apply_differential(our_centres, bg_traj), lds)


if __name__ == '__main__':
    main()
