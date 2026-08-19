"""
Regression check: re-run the template-match + affine-flow configuration
end-to-end and confirm it still reproduces its recorded 0.07039 mm RMSE /
0.98290 correlation exactly.

templatematch_affine_config was the first configuration that worked on this dataset. It is kept
verbatim and driven from here rather than being merged into viv-bench, for two
reasons:

  1. It is the project's only *fixed point*. Every refactor, bug fix and new
     method in viv-bench is validated by confirming this number does not move.
     Two separate measurement bugs (see docs/report.html) were caught because
     this reproduction stayed put while other results shifted.
  2. Re-running it from the raw video - rather than re-reading its saved CSV -
     means the check covers the whole chain: decode, target tracking,
     background-feature affine compensation, filtering and LDS comparison.

The one substitution is the interactive ROI step: templatematch_affine_config normally opens a
cv2.selectROI GUI, so this driver supplies the same ROI values from its saved
templatematch_affine_config_json.json to make the run unattended. Its functions are loaded
unmodified via importlib (the original filename contains parentheses and so is
not importable as a module name).

Usage:
    python experiments/run_templatematch_affine_config.py
"""
import importlib.util
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import video_path, lds_path, templatematch_affine_script, templatematch_affine_config_json

TM_AFFINE_SCRIPT = templatematch_affine_script()
TM_AFFINE_CONFIG = templatematch_affine_config_json()
VIDEO_PATH = video_path()
LDS_PATH = lds_path()


def load_config_module(path):
    spec = importlib.util.spec_from_file_location("tm_affine_config_module", path)
    module = importlib.util.module_from_spec(spec)
    # Must be registered in sys.modules before exec: the script uses
    # @dataclass with `from __future__ import annotations`, and dataclass's
    # forward-reference resolution looks the module up via
    # sys.modules.get(cls.__module__) - which is None (crashes) otherwise.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    sb = load_config_module(TM_AFFINE_SCRIPT)

    with open(TM_AFFINE_CONFIG) as f:
        config = json.load(f)

    target_roi = tuple(config['target_roi'])
    stab_rois = [tuple(r) for r in config['stab_rois']]
    mm_per_px = config['mm_per_px']
    u_normal = np.array(config['u_normal'], dtype=np.float64)
    search_margin = config.get('search_margin', 70)
    highpass_hz = config.get('highpass_hz', 0.20)
    lowpass_hz = config.get('lowpass_hz', 20.0)

    print("=" * 70)
    print("REGRESSION CHECK: frozen template-match + affine-flow config, unmodified (ROIs from its saved settings)")
    print("=" * 70)
    print(f"target_roi={target_roi}  stab_rois={stab_rois}")
    print(f"mm_per_px={mm_per_px:.6f}  u_normal={u_normal}  search_margin={search_margin}")
    print(f"filter: highpass={highpass_hz}Hz lowpass={lowpass_hz}Hz")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame")
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    target_state = sb.make_target_state(first, target_roi)
    pts0 = sb.build_static_feature_points(first_gray, stab_rois)
    print(f"Static feature points: {len(pts0)}")

    rows = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        c_img, c_score, c_ok = sb.track_target_template(gray, target_state, search_margin)
        M, n_inliers, flow_err = sb.estimate_camera_affine(first_gray, gray, pts0)
        c_stab = sb.apply_inverse_affine(M, c_img)
        d_vec_stab = c_stab - target_state.center0
        disp_px = float(np.dot(d_vec_stab, u_normal))
        disp_mm = disp_px * mm_per_px

        rows.append({
            "frame": frame_idx,
            "time_s": frame_idx / fps,
            "displacement_mm_motion_compensated": disp_mm,
            "target_match_score": c_score,
            "static_feature_inliers": n_inliers,
        })

        if frame_idx % 500 == 0:
            print(f"  processed frame {frame_idx}/{n_frames_total}")
        frame_idx += 1
    cap.release()

    df = pd.DataFrame(rows)
    y0 = df["displacement_mm_motion_compensated"].to_numpy(np.float64)
    y0 = y0 - np.nanmean(y0)
    y_proc = sb.filter_signal(y0, fs=fps, highpass_hz=highpass_hz, lowpass_hz=lowpass_hz)

    submission_y = y_proc - np.nanmean(y_proc)

    t_lds, y_lds = sb.read_lds_xlsx(LDS_PATH, lds_fs=10000.0)
    y_lds_proc = sb.filter_signal(y_lds, fs=10000.0, highpass_hz=highpass_hz, lowpass_hz=lowpass_hz)
    best = sb.compare_with_lds(
        df["time_s"].to_numpy(np.float64), submission_y, t_lds, y_lds_proc, fps=fps, max_lag_s=1.0
    )

    print("\n" + "=" * 70)
    print("REPRODUCTION RESULT")
    print("=" * 70)
    print(json.dumps(best, indent=2))
    print(f"\nRecorded in the template-match + affine-flow configuration's own lds_alignment_v2.json: rmse_mm=0.07039  corr=0.98290")
    print(f"Reproduced here:                                             rmse_mm={best['rmse_mm']:.5f}  corr={best['corr']:.5f}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'templatematch_affine_config')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'reproduction_result.json'), 'w') as f:
        json.dump(best, f, indent=2)
    df.to_csv(os.path.join(out_dir, 'tracking_full.csv'), index=False)
    print(f"\nSaved to {out_dir}")


if __name__ == '__main__':
    main()
