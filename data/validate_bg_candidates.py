"""
Propose and empirically validate wider-spread background ROIs.

Motivation (see README "Literature survey"): ego-motion error is minimised by
expanding the effective field of view of the reference features and sampling
motion from opposite directions. The two ROIs shipped in configs/rois.json sit
at (1354,811) and (2472,1303) of a 3840x2160 frame - only ~31% of the width,
both mid-frame, and both on free-standing tripods, which are not obviously
rigid. The 0.2-1Hz drift band is the largest remaining error term, and that is
exactly what a better-conditioned ego-motion estimate should reduce.

A candidate box is only useful if it is (a) textured enough to yield features,
(b) genuinely static - i.e. its motion is explained by camera motion alone.
Eyeballing the frame cannot establish (b), so each candidate is scored here:
its tracked motion is regressed against the incumbent background points (a
proxy for true camera motion) and the unexplained residual is reported.
A rigid, static region should show a small residual; a compliant or
independently-moving region (a swaying tripod, the cable, a cable-anchoring
frame) shows a large one.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import video_path

VIDEO = video_path()

# Hand-proposed from a visual inspection of frame 0, chosen to be rigid
# building/rig structure (not tripods, not the cable) and spread far wider
# than the incumbent pair. Full-frame pixel coords, (x, y, w, h).
CANDIDATES = {
    'incumbent_tripod_L':  (1354, 811, 67, 67),     # currently used
    'incumbent_tripod_R':  (2472, 1303, 67, 70),    # currently used
    'ceiling_light_TL':    (880, 60, 220, 130),     # ceiling fixture, top-left
    'wall_seam_L':         (180, 1000, 420, 200),   # left wall panel seams
    'floor_plates_BL':     (300, 1680, 900, 400),   # floor plate texture, bottom-left
    'window_frames_R':     (2580, 450, 660, 600),   # window mullions + life ring, right
    'blue_frame_TR':       (3400, 60, 420, 500),    # structural steel, top-right
    'blue_frame_BR':       (3390, 1440, 440, 640),  # structural steel, bottom-right
    'machinery_R':         (3150, 1480, 560, 260),  # rig hardware, right
}


def detect_box_features(first_gray, box, max_corners=250):
    x, y, w, h = box
    roi = first_gray[y:y + h, x:x + w]
    pts = cv2.goodFeaturesToTrack(roi, maxCorners=max_corners, qualityLevel=0.01,
                                   minDistance=10, blockSize=7, useHarrisDetector=False)
    if pts is None:
        return np.zeros((0, 2), np.float32)
    pts = pts.reshape(-1, 2)
    pts[:, 0] += x
    pts[:, 1] += y
    return pts.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=600)
    ap.add_argument('--stride', type=int, default=2)
    args = ap.parse_args()

    cap = cv2.VideoCapture(VIDEO)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError('cannot read video')
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    frames = []
    idx = 0
    while len(frames) < args.frames // args.stride:
        ok, f = cap.read()
        if not ok:
            break
        if idx % args.stride == 0:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        idx += 1
    cap.release()
    print(f"tracking {len(frames)} frames (stride {args.stride})\n")

    # Detect features per box, pool them, and track the pooled set once.
    names, boxes, pts_list, owner = [], [], [], []
    for i, (name, box) in enumerate(CANDIDATES.items()):
        p = detect_box_features(first_gray, box)
        print(f"  {name:22s} features={len(p):4d}")
        if len(p) == 0:
            continue
        names.append(name); boxes.append(box); pts_list.append(p)
        owner.append(np.full(len(p), len(names) - 1))
    pts0 = np.vstack(pts_list)
    owner = np.concatenate(owner)
    p0 = pts0.reshape(-1, 1, 2)

    # Per frame: fit ONE global similarity transform across all candidate
    # features (RANSAC), then measure each box's own reprojection residual
    # against it. A rigid static region agrees with the global rigid motion;
    # a region that moves independently (compliant mount, the cable itself)
    # shows up as a consistent outlier. Regressing against a single reference
    # box cannot do this - camera rotation/zoom alone makes different image
    # positions translate by genuinely different amounts.
    per_box_resid = [[] for _ in names]
    for g in frames:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            first_gray, g, p0, None, winSize=(31, 31), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if p1 is None or st is None:
            continue
        m = st.reshape(-1).astype(bool)
        if m.sum() < 20:
            continue
        a, b, own = pts0[m], p1.reshape(-1, 2)[m], owner[m]
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                              ransacReprojThreshold=3.0,
                                              maxIters=2000, confidence=0.99)
        if M is None:
            continue
        pred = (np.hstack([a, np.ones((len(a), 1), np.float32)]) @ M.T)
        err = np.linalg.norm(b - pred, axis=1)
        for i in range(len(names)):
            sel = own == i
            if sel.any():
                per_box_resid[i].append(float(np.median(err[sel])))

    print(f"\n{'candidate':22s} {'feat':>5} {'reproj_resid_px':>16}  verdict")
    scored = []
    for i, name in enumerate(names):
        r = float(np.median(per_box_resid[i])) if per_box_resid[i] else np.nan
        scored.append((r, name, boxes[i], len(pts_list[i])))
    for r, name, box, nf in scored:
        verdict = ('GOOD (rigid, agrees with global motion)' if r < 0.5 else
                   'marginal' if r < 1.0 else 'REJECT (inconsistent / moves independently)')
        print(f"{name:22s} {nf:5d} {r:16.3f}  {verdict}")

    print("\nreproj_resid_px = median distance from the globally-fitted rigid motion.")
    print("Low = the region is rigid and static, and belongs in the reference set.")


if __name__ == '__main__':
    main()
