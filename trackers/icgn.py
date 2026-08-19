"""
Inverse-Compositional Gauss-Newton (IC-GN) sub-pixel tracker (Tier-1).

The DIC (Digital Image Correlation) field's standard sub-pixel registration
technique - flagged in the original project plan as "the structural-
engineering gold standard" but never implemented until now. Unlike
NCC-peak/phase-correlation trackers (which locate a single correlation peak
and then interpolate around it), IC-GN directly minimizes sum-of-squared
intensity differences between the template and the warped current frame
using every pixel's gradient information, via Newton iteration.

Reference: Baker & Matthews, "Lucas-Kanade 20 Years On: A Unifying
Framework" (2004) - inverse-compositional algorithm, specialized here to a
pure-translation warp (2 DOF), which is what a small rigid target
undergoing sub-pixel motion needs.

Algorithm per frame (warm-started from the previous frame's converged p):
  1. (Precomputed once from the frame-0 template T:)
     - Template gradient (Tx, Ty)
     - Steepest-descent images = [Tx, Ty] (Jacobian of a translation warp is
       the identity, so no chain rule needed)
     - Hessian H = sum_x [Tx,Ty]^T [Tx,Ty], inverted once
  2. Warp the current frame by the current estimate p=(dx,dy) (bilinear
     sub-pixel sampling), compute error = I(W(x;p)) - T(x)
  3. delta_p = H^-1 * sum_x [Tx,Ty]^T * error(x)
  4. p <- p - delta_p (inverse-compositional update; for pure translation,
     composing with the inverse of a small warp is just subtraction)
  5. Repeat until |delta_p| < tol or max_iters reached
"""
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trackers.base import Tracker


class ICGNTracker(Tracker):
    """Inverse-compositional Gauss-Newton tracker (the DIC standard).

        Iteratively warps the frame-0 template onto each later frame, solving for
        the warp update in the template frame so the Hessian is computed once
        instead of per iteration. Sub-pixel sampling uses cubic interpolation.

        Args:
            template_size: side length of the square template taken from frame 0
            interp_order: spline order for warp sampling (3 = cubic)
        """
    def __init__(self, name='IC-GN', template_size=64, max_iters=20, tol=1e-4,
                 interp_order=1):
        super().__init__(name, max_frames=None)
        self.template_size = template_size
        self.max_iters = max_iters
        self.tol = tol
        # DIC literature: sub-pixel peak/warp estimates are systematically
        # biased toward integer pixels ("peak-locking"), worse for
        # high-spatial-frequency content like a checkerboard's sharp edges,
        # and reduced by higher-order (cubic) interpolation vs. bilinear.
        self.interp_order = interp_order

    def _precompute(self, template):
        template = template.astype(np.float64)
        # Central-difference gradient, consistent with the bilinear sampling
        # used when warping the current frame.
        Ty, Tx = np.gradient(template)

        H = np.array([
            [np.sum(Tx * Tx), np.sum(Tx * Ty)],
            [np.sum(Tx * Ty), np.sum(Ty * Ty)],
        ])
        H_inv = np.linalg.pinv(H)
        return template, Tx, Ty, H_inv

    def track(self, frames, query_points):
        """Track the query point across all frames.

                Args:
                    frames: (T, H, W) uint8 patch stack
                    query_points: (1, 2) start position in patch-local coordinates

                Returns:
                    trajectories: (T, 1, 2) tracked position per frame
                    confidence: (T,) per-frame convergence/quality measure
                    meta: dict of tracker metadata
                """
        T_len, H, W = frames.shape
        N = len(query_points)
        hs = self.template_size // 2

        trajectories = np.zeros((T_len, N, 2), dtype=np.float32)
        confidence = np.zeros((T_len, N), dtype=np.float32)
        trajectories[0, :, :] = query_points
        confidence[0, :] = 1.0

        frame0 = frames[0]
        precomputed = []
        for n in range(N):
            x, y = int(round(query_points[n, 0])), int(round(query_points[n, 1]))
            y0, y1 = max(0, y - hs), min(H, y + hs)
            x0, x1 = max(0, x - hs), min(W, x + hs)
            template = frame0[y0:y1, x0:x1]

            # Coordinates (row, col) of the template's pixels within the
            # frame, at the reference (p=0) position.
            rows = np.arange(y0, y1, dtype=np.float64)
            cols = np.arange(x0, x1, dtype=np.float64)
            grid_r, grid_c = np.meshgrid(rows, cols, indexing='ij')

            if template.size == 0:
                precomputed.append(None)
                continue

            T_arr, Tx, Ty, H_inv = self._precompute(template)
            template_norm = float(np.std(T_arr)) + 1e-10
            precomputed.append({
                'T': T_arr, 'Tx': Tx, 'Ty': Ty, 'H_inv': H_inv,
                'grid_r': grid_r, 'grid_c': grid_c, 'norm': template_norm,
            })

        p = np.zeros((N, 2), dtype=np.float64)  # (dx, dy) per point, warm-started

        for t in range(1, T_len):
            frame_t = frames[t].astype(np.float64)

            for n in range(N):
                pc = precomputed[n]
                if pc is None:
                    trajectories[t, n, :] = trajectories[t - 1, n, :]
                    confidence[t, n] = 0.0
                    continue

                dx, dy = p[n]
                converged_err = None

                for _ in range(self.max_iters):
                    sample_r = pc['grid_r'] + dy
                    sample_c = pc['grid_c'] + dx
                    I_warped = ndimage.map_coordinates(
                        frame_t, [sample_r, sample_c], order=self.interp_order, mode='nearest'
                    )
                    error = I_warped - pc['T']

                    b = np.array([
                        np.sum(pc['Tx'] * error),
                        np.sum(pc['Ty'] * error),
                    ])
                    delta_p = pc['H_inv'] @ b
                    dx -= delta_p[0]
                    dy -= delta_p[1]

                    converged_err = error
                    if np.hypot(delta_p[0], delta_p[1]) < self.tol:
                        break

                p[n] = (dx, dy)
                trajectories[t, n, 0] = query_points[n, 0] + dx
                trajectories[t, n, 1] = query_points[n, 1] + dy

                rms_err = float(np.sqrt(np.mean(converged_err ** 2))) if converged_err is not None else 0.0
                confidence[t, n] = float(np.clip(1.0 - rms_err / (pc['norm'] * 3.0), 0.0, 1.0))

        meta = {
            'tracker': self.name,
            'template_size': self.template_size,
            'max_iters': self.max_iters,
        }
        return trajectories, confidence, meta
