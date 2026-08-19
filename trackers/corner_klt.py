"""
Corner-KLT-averaged tracker (Tier-1 classical method, from the original plan).

Detects several sub-pixel-refined corners within the target ROI (frame 0),
tracks each independently with pyramidal Lucas-Kanade optical flow anchored
to frame 0 (not frame-to-frame, to avoid drift accumulation - same anchoring
choice as the other trackers here), and averages the resulting displacement
across corners. Averaging over ~N independent noisy corner estimates
suppresses per-corner tracking noise by roughly sqrt(N) if the noise is
uncorrelated - a different noise-reduction mechanism than IC-GN's dense
whole-template optimization or the phase-based tracker's frequency-domain
approach, so it's expected to fail differently (useful as an independent
comparison point) even if it doesn't beat them outright.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trackers.base import Tracker


class CornerKLTTracker(Tracker):
    """Averaged corner + pyramidal Lucas-Kanade tracker.

        Rather than tracking one point, detects many corners inside the target
        template and averages their optical-flow displacement. Averaging N corners
        reduces per-corner noise, which is why this is the best-performing target
        tracker in the ablation - but note it degrades on low-texture patches (see
        README: corner_klt + differential is the one bad tracker/compensator pair).
        """
    def __init__(self, name='Corner-KLT', template_size=64, max_corners=25,
                 quality_level=0.05, min_distance=5, win_size=21, max_level=3):
        super().__init__(name, max_frames=None)
        self.template_size = template_size
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.win_size = win_size
        self.max_level = max_level

    def track(self, frames, query_points):
        """Track the query point across all frames.

                Returns (trajectories (T,1,2), confidence (T,), meta) - the shared
                Tracker contract; see trackers/base.py.
                """
        T, H, W = frames.shape
        N = len(query_points)
        hs = self.template_size // 2

        trajectories = np.zeros((T, N, 2), dtype=np.float32)
        confidence = np.zeros((T, N), dtype=np.float32)
        trajectories[0, :, :] = query_points
        confidence[0, :] = 1.0

        frame0 = frames[0]
        corners_list = []
        offsets = []

        for n in range(N):
            x, y = int(round(query_points[n, 0])), int(round(query_points[n, 1]))
            y0, y1 = max(0, y - hs), min(H, y + hs)
            x0, x1 = max(0, x - hs), min(W, x + hs)
            roi0 = frame0[y0:y1, x0:x1]

            corners = cv2.goodFeaturesToTrack(
                roi0, maxCorners=self.max_corners, qualityLevel=self.quality_level,
                minDistance=self.min_distance, blockSize=5, useHarrisDetector=False,
            )
            if corners is None or len(corners) < 3:
                corners_list.append(None)
                offsets.append((x0, y0))
                continue

            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            cv2.cornerSubPix(roi0, corners, (5, 5), (-1, -1), criteria)
            corners_full = corners.reshape(-1, 2) + np.array([x0, y0])
            corners_list.append(corners_full.astype(np.float32).reshape(-1, 1, 2))
            offsets.append((x0, y0))

        for t in range(1, T):
            frame_t = frames[t]

            for n in range(N):
                corners0 = corners_list[n]
                if corners0 is None:
                    trajectories[t, n, :] = trajectories[t - 1, n, :]
                    confidence[t, n] = 0.0
                    continue

                pts1, st, err = cv2.calcOpticalFlowPyrLK(
                    frame0, frame_t, corners0, None,
                    winSize=(self.win_size, self.win_size), maxLevel=self.max_level,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )
                if pts1 is None or st is None or not st.any():
                    trajectories[t, n, :] = trajectories[t - 1, n, :]
                    confidence[t, n] = 0.0
                    continue

                st_mask = st.reshape(-1).astype(bool)
                disp = pts1.reshape(-1, 2)[st_mask] - corners0.reshape(-1, 2)[st_mask]
                mean_disp = disp.mean(axis=0)

                trajectories[t, n, :] = query_points[n] + mean_disp
                confidence[t, n] = float(st_mask.mean())

        meta = {
            'tracker': self.name,
            'template_size': self.template_size,
            'max_corners': self.max_corners,
        }
        return trajectories, confidence, meta
