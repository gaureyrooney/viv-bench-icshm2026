"""
Why does every method plateau at ~0.070mm? Direct measurement of the noise
budget, rather than inference from aggregate RMSE.

Established already:
  - LDS sensor noise is ~0.00002mm in-band (PSD floor 1.6e-11 mm^2/Hz above
    100Hz) - 3500x below the plateau, so the reference is NOT the limit.
  - At resonance the vision/LDS gain is 0.989-0.999 and removing gain error
    explains 0% of the resonance-band residual - so the plateau is not a
    scale/amplitude error either.
  - Residual splits as ~0.038mm inside 10.5-13Hz and ~0.060mm outside it,
    i.e. ~72% of residual energy is vision content in bands where the LDS
    shows almost nothing => broadband measurement noise.

This script attributes that broadband noise between the two candidate
sources, by resampling the estimator inputs:

  (A) COMPENSATION noise. The per-frame background transform is fitted from
      only ~17 features. Split them into two disjoint halves, fit an
      independent transform from each, compensate the SAME target track with
      both, and difference the results. Two independent half-set estimates
      differ by sqrt(2) x the half-set noise, and a half-set fit is ~sqrt(2)
      noisier than the full-set fit, so
          sigma_full ~= rms(difference) / 2.
      This needs no ground truth - it measures the estimator's own variance.

  (B) TARGET-TRACKING noise. Bounded from the disagreement between two
      independent trackers (icgn vs corner_klt) run on identical frames with
      identical compensation; their difference removes all shared terms.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import video_path, lds_path, templatematch_affine_script, templatematch_affine_config_json
from dsp.alignment import zero_phase_filter

VIDEO = video_path()
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
BG_BOXES = [(1354, 811, 67, 67), (2472, 1303, 67, 70)]
TARGET_XY = np.array([1899.5, 888.0])
U_VIV = np.array([0.45016305, -0.89294638])
MM_PER_PX = 0.8661417322834646
CACHE = os.path.join(RES, 'noise_budget', 'bg_flow.npz')


def collect_bg_flow(max_frames=None):
    """Per-frame tracked positions of every background feature (one pass)."""
    cap = cv2.VideoCapture(VIDEO)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError('cannot read video')
    fg = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    pts = []
    for (x, y, w, h) in BG_BOXES:
        c = cv2.goodFeaturesToTrack(fg[y:y + h, x:x + w], maxCorners=250,
                                    qualityLevel=0.01, minDistance=10, blockSize=7,
                                    useHarrisDetector=False)
        if c is None:
            continue
        c = c.reshape(-1, 2)
        c[:, 0] += x
        c[:, 1] += y
        pts.append(c)
    pts0 = np.vstack(pts).astype(np.float32)
    print(f"    {len(pts0)} background features")

    tracked, status = [], []
    t = 0
    while True:
        if max_frames is not None and t >= max_frames:
            break
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            fg, g, pts0.reshape(-1, 1, 2), None, winSize=(31, 31), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        tracked.append(p1.reshape(-1, 2) if p1 is not None else np.full_like(pts0, np.nan))
        status.append(st.reshape(-1).astype(bool) if st is not None else np.zeros(len(pts0), bool))
        if t % 500 == 0:
            print(f"    frame {t}")
        t += 1
    cap.release()
    return pts0, np.array(tracked), np.array(status)


def compensate_with_subset(pts0, tracked, status, idx):
    """Fit a similarity transform per frame using only features `idx`, then
    map the fixed target point through its inverse - the same operation
    AffineFlowCompensator.compensate_point performs."""
    out = np.zeros((len(tracked), 2))
    for t in range(len(tracked)):
        m = status[t][idx]
        a, b = pts0[idx][m], tracked[t][idx][m]
        if len(a) < 3:
            out[t] = out[t - 1] if t else TARGET_XY
            continue
        M, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                           ransacReprojThreshold=3.0,
                                           maxIters=2000, confidence=0.99)
        if M is None:
            out[t] = out[t - 1] if t else TARGET_XY
            continue
        Minv = cv2.invertAffineTransform(M)
        x, y = TARGET_XY
        out[t] = [Minv[0, 0] * x + Minv[0, 1] * y + Minv[0, 2],
                  Minv[1, 0] * x + Minv[1, 1] * y + Minv[1, 2]]
    return out


def to_mm(traj):
    return (traj - traj[0]) @ U_VIV * MM_PER_PX


def main():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        pts0, tracked, status = z['pts0'], z['tracked'], z['status']
        print(f"[cache] {len(pts0)} features, {len(tracked)} frames")
    else:
        print("[run] one video pass collecting background feature flow...")
        pts0, tracked, status = collect_bg_flow()
        np.savez(CACHE, pts0=pts0, tracked=tracked, status=status)

    n = len(pts0)
    rng = np.random.default_rng(0)
    bp = lambda x: zero_phase_filter(x, fs=50, cutoff_hz=(0.2, 20.), btype='band')

    print("\n=== (A) COMPENSATION noise, from disjoint half-set refits ===")
    diffs = []
    for trial in range(5):
        perm = rng.permutation(n)
        A, B = perm[:n // 2], perm[n // 2:]
        ca = to_mm(compensate_with_subset(pts0, tracked, status, A))
        cb = to_mm(compensate_with_subset(pts0, tracked, status, B))
        d = bp(ca - cb)
        diffs.append(np.sqrt(np.mean(d ** 2)))
        print(f"  trial {trial}: rms(half-A - half-B) = {diffs[-1]:.5f} mm")
    dm = float(np.mean(diffs))
    print(f"  mean = {dm:.5f} mm  =>  full-set compensation noise ~= {dm / 2:.5f} mm")

    print("\n=== (B) TARGET-TRACKING noise, from tracker disagreement ===")
    try:
        a = np.load(os.path.join(RES, 'postaudit_cornerklt_affineflow', 'aligned_signals.npz'))['displacement_mm']
        b = np.load(os.path.join(RES, 'icgn_affineflow_bp0.2-20_symlds_SIGNFIX', 'aligned_signals.npz'))['displacement_mm']
        m = min(len(a), len(b))
        d = bp(a[:m] - b[:m])
        rms = float(np.sqrt(np.mean(d ** 2)))
        print(f"  rms(corner_klt - icgn), same compensation = {rms:.5f} mm")
        print(f"  => per-tracker noise ~= {rms / np.sqrt(2):.5f} mm (if independent)")
    except FileNotFoundError as e:
        print(f"  missing run: {e}")

    print("\n=== BUDGET vs the observed plateau ===")
    obs = 0.0707
    comp = dm / 2
    print(f"  observed residual                 {obs:.5f} mm")
    print(f"  compensation estimator noise (A)  {comp:.5f} mm  ({100 * comp**2 / obs**2:.0f}% of variance)")
    print(f"  LDS sensor noise                  0.00002 mm  (negligible)")
    print(f"  unexplained remainder             {np.sqrt(max(obs**2 - comp**2, 0)):.5f} mm")


if __name__ == '__main__':
    main()
