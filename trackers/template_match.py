"""
Sub-pixel template-matching tracker (Tier-1 classical method).

Normalized cross-correlation (cv2.matchTemplate, TM_CCOEFF_NORMED) against a
fixed frame-0 template, with CLAHE contrast normalization and parabolic
sub-pixel peak interpolation. This mirrors a proven approach (0.070 mm RMSE,
0.983 correlation on this dataset) ratits than inventing a new one - see
the template-match + affine-flow configuration's icshm2026_project1_templatematch_affine_config.py, track_target_template().
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trackers.base import Tracker


def subpixel_offset(res, x, y):
    """Quadratic (parabolic) peak interpolation around the integer peak of a
    cv2.matchTemplate response map, independently in x and y."""
    dx = 0.0
    dy = 0.0
    if 1 <= x < res.shape[1] - 1:
        left, mid, right = float(res[y, x - 1]), float(res[y, x]), float(res[y, x + 1])
        denom = left - 2.0 * mid + right
        if abs(denom) > 1e-12:
            dx = float(np.clip(0.5 * (left - right) / denom, -1.0, 1.0))
    if 1 <= y < res.shape[0] - 1:
        up, mid, down = float(res[y - 1, x]), float(res[y, x]), float(res[y + 1, x])
        denom = up - 2.0 * mid + down
        if abs(denom) > 1e-12:
            dy = float(np.clip(0.5 * (up - down) / denom, -1.0, 1.0))
    return dx, dy


class TemplateMatchTracker(Tracker):
    """
    Tracks each query point by matching a fixed frame-0 template against a
    search window around the previous position, every frame.
    """

    def __init__(self, name='Template-Match', template_size=32, search_margin=50,
                 use_clahe=True, min_score=0.35):
        super().__init__(name, max_frames=None)
        self.template_size = template_size
        self.search_margin = search_margin
        self.use_clahe = use_clahe
        self.min_score = min_score
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None

    def _preprocess(self, frame):
        if self._clahe is not None:
            return self._clahe.apply(frame)
        return frame

    def track(self, frames, query_points):
        """Track the query point across all frames. See trackers/base.py."""
        T, H, W = frames.shape
        N = len(query_points)
        hs = self.template_size // 2

        trajectories = np.zeros((T, N, 2), dtype=np.float32)
        confidence = np.zeros((T, N), dtype=np.float32)
        trajectories[0, :, :] = query_points
        confidence[0, :] = 1.0

        frame0 = self._preprocess(frames[0])
        templates = []
        for n in range(N):
            x, y = int(round(query_points[n, 0])), int(round(query_points[n, 1]))
            y0, y1 = max(0, y - hs), min(H, y + hs)
            x0, x1 = max(0, x - hs), min(W, x + hs)
            templates.append(frame0[y0:y1, x0:x1].copy())

        center_prev = query_points.copy().astype(np.float64)

        for t in range(1, T):
            frame_t = self._preprocess(frames[t])

            for n in range(N):
                templ = templates[n]
                if templ.size == 0 or templ.shape[0] < 4 or templ.shape[1] < 4:
                    trajectories[t, n, :] = trajectories[t - 1, n, :]
                    confidence[t, n] = 0.0
                    continue

                cx, cy = center_prev[n]
                th, tw = templ.shape
                sx0 = max(0, int(round(cx - tw / 2 - self.search_margin)))
                sy0 = max(0, int(round(cy - th / 2 - self.search_margin)))
                sx1 = min(W, int(round(cx + tw / 2 + self.search_margin)))
                sy1 = min(H, int(round(cy + th / 2 + self.search_margin)))
                search = frame_t[sy0:sy1, sx0:sx1]

                if search.shape[0] < th or search.shape[1] < tw:
                    trajectories[t, n, :] = trajectories[t - 1, n, :]
                    confidence[t, n] = 0.0
                    continue

                res = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                px, py = max_loc
                dx, dy = subpixel_offset(res, px, py)

                top_left_x = sx0 + px + dx
                top_left_y = sy0 + py + dy
                center = np.array([top_left_x + tw / 2.0, top_left_y + th / 2.0])

                trajectories[t, n, :] = center
                confidence[t, n] = max_val

                if max_val > self.min_score:
                    center_prev[n] = center
                # else: keep searching around the last good position next frame

        meta = {
            'tracker': self.name,
            'template_size': self.template_size,
            'search_margin': self.search_margin,
        }
        return trajectories, confidence, meta
