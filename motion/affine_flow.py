"""
Ego-motion compensation via many background feature points.

Unlike DifferentialReferenceCompensator (which mean-subtracts the position of
a couple of tracked background *points* - translation only), this detects
many corner features inside each background ROI, tracks them with pyramidal
Lucas-Kanade optical flow, and robustly fits a per-frame global transform
(RANSAC-filtered). That lets it correct camera rotation/scale drift that pure
mean-translation can't, which is what the template-match + affine-flow configuration uses to reach 0.070 mm RMSE (vs.
~0.15 mm with 2-point mean-translation compensation).

motion_model (exposed as --affine-model {similarity,full,homography} in
run_experiment.py, via experiments/registries.py's AFFINE_MODEL_TO_MOTION_MODEL):
  'affine' (default; CLI: 'similarity') - 4 DOF similarity transform
      (translation+rotation+uniform scale, cv2.estimateAffinePartial2D). This
      is what templatematch_affine_config uses, error-percentile pre-filter included
      (err_keep_pct), and it beats the richer models by 2.7x here.
  'full_affine' (CLI: 'full') - 6 DOF affine (adds independent x/y scale +
      shear, cv2.estimateAffine2D). Not used by templatematch_affine_config; included for
      comparison only - measured 0.1931 mm vs 0.0711 mm for similarity.
  'homography' - full projective transform (8 DOF, cv2.findHomography). Several
      2024-2026 UAV structural-monitoring papers prefer this over affine
      specifically because it also captures perspective effects from small
      UAV attitude/depth changes that a 2D affine can't represent, without
      needing camera intrinsics. Tested here: underperformed with only ~2
      small/clustered background ROIs (see README).

Because it needs pixel data (to detect/track corners), not just already-
tracked point trajectories, this does NOT implement the same fit(traj_target,
traj_bg) interface as motion/differential_ref.py's compensators - it operates
on raw background ROI patches instead. pipeline.py dispatches accordingly.

Optional temporal smoothing (transform_smoothing={'none','ema','kalman'}):
each frame's transform above is fit independently by RANSAC, with no
coupling between frames - so any single frame's noisy corner set (motion
blur, a few bad matches slipping past the error-percentile filter) shows up
directly as noise in the compensated target trajectory. Smoothing couples
neighboring frames' fits together to average that out, *if* the true
camera motion varies slowly relative to the 50Hz frame rate - which is an
assumption to verify per-run (see get_transform_debug_arrays), not assume:
over-smoothing would suppress genuine fast camera motion (e.g. if the UAV
airframe itself picks up some of the cable's ~11.7Hz resonance) exactly as
readily as it suppresses per-frame RANSAC noise, and the two are not
distinguishable from the transform alone.
"""
import cv2
import numpy as np


# --------------------------------------------------------------------------
# Transform <-> (translation, rotation, scale_x, scale_y, shear) decomposition
#
# Smoothing raw 2x3 matrix entries directly can produce non-rigid
# interpolation artifacts (e.g. averaging two rotation matrices doesn't give
# a rotation matrix), so smoothing operates on this decomposition instead,
# then recomposes. One general 6-parameter decomposition covers both
# 'affine' (similarity: 4-DOF, RANSAC already constrains scale_x==scale_y
# and shear==0, so those two params just naturally track each other and
# ~0) and 'full_affine' (6-DOF) - not homography, whose row-3 projective
# terms this decomposition doesn't represent (see AffineFlowCompensator's
# constructor guard).
#
# Decomposition follows the standard `M = R(theta) @ Shear(shear) @
# Scale(scale_x, scale_y)` factorization (the same one browsers use for CSS
# `matrix()` transforms) - see PARAM_NAMES ordering below.
# --------------------------------------------------------------------------

PARAM_NAMES = ('tx', 'ty', 'theta', 'scale_x', 'scale_y', 'shear')


def decompose_transform(M3):
    """(3,3) homogeneous matrix -> (tx, ty, theta_rad, scale_x, scale_y, shear)."""
    a, b, tx = M3[0]
    c, d, ty = M3[1]
    scale_x = float(np.hypot(a, c))
    theta = float(np.arctan2(c, a))
    ct, st = np.cos(theta), np.sin(theta)
    scale_y = float(d * ct - b * st)
    shear = float((b * ct + d * st) / scale_y) if abs(scale_y) > 1e-12 else 0.0
    return np.array([tx, ty, theta, scale_x, scale_y, shear], dtype=np.float64)


def compose_transform(params):
    """Inverse of decompose_transform - round-trips exactly (see
    tests/synthetic validation): (tx, ty, theta, scale_x, scale_y, shear) ->
    (3,3) homogeneous matrix."""
    tx, ty, theta, scale_x, scale_y, shear = params
    ct, st = np.cos(theta), np.sin(theta)
    a = scale_x * ct
    c = scale_x * st
    b = scale_y * (shear * ct - st)
    d = scale_y * (shear * st + ct)
    return np.array([[a, b, tx], [c, d, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def decompose_transform_sequence(transform_list):
    """List of (3,3) matrices -> (T, 6) param array, with theta unwrapped
    across the sequence (frame-to-frame UAV jitter rotation is small and
    never genuinely wraps a full turn; unwrapping avoids a spurious +/-2pi
    jump being treated as real motion by the smoothers below)."""
    params = np.array([decompose_transform(M) for M in transform_list])
    params[:, 2] = np.unwrap(params[:, 2])
    return params


def compose_transform_sequence(params):
    """Inverse of decompose_transform_sequence: (T, 6) params -> list of (3,3)."""
    return [compose_transform(row) for row in params]


def ema_smooth(params, alpha):
    """Exponential moving average, per parameter column independently.
    alpha is the weight on the new (raw) sample each step - alpha=1 reduces
    to no smoothing (matches raw exactly); smaller alpha smooths more
    heavily. smoothed[0] = raw[0] (nothing to average against yet)."""
    smoothed = np.empty_like(params)
    smoothed[0] = params[0]
    for t in range(1, len(params)):
        smoothed[t] = alpha * params[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def _measurement_noise_scale(n_inliers, flow_err):
    """Per-frame multiplicative scale on Kalman measurement variance R, from
    the two rough proxies requested: fewer inliers or higher LK error ->
    trust this frame's fit less (bigger scale -> bigger R -> the filter
    leans more on its own constant-velocity prediction instead)."""
    n_inliers = np.asarray(n_inliers, dtype=np.float64)
    flow_err = np.asarray(flow_err, dtype=np.float64)
    typical_n = np.nanmedian(n_inliers[n_inliers > 0]) if np.any(n_inliers > 0) else 1.0
    finite_err = flow_err[np.isfinite(flow_err)]
    typical_err = np.nanmedian(finite_err) if len(finite_err) else 0.0

    inlier_factor = typical_n / np.maximum(n_inliers, 1.0)
    err_safe = np.where(np.isfinite(flow_err), flow_err, typical_err)
    err_factor = 1.0 + err_safe / (typical_err + 1e-9)
    return inlier_factor * err_factor


def kalman_smooth_1d(z, r_scale, q_ratio=1.0, dt=1.0):
    """Constant-velocity Kalman filter (position + velocity state) over one
    scalar parameter sequence, RTS-smoothed (forward filter + backward pass)
    so early frames benefit from later measurements too, not just a causal
    filter's warm-up transient.

    Process noise uses the standard discretized white-noise-acceleration
    model, sized from the sequence's own frame-to-frame variability
    (q_ratio * var(diff(z))) rather than a fixed constant that would be
    right for pixel-scale translation params and wildly wrong for
    near-1.0 scale or near-0 radian params.

    IMPORTANT, found by synthetic testing (see README "transform smoothing"):
    a constant-velocity model is a structurally poor fit for a genuinely
    oscillating true signal (its velocity reverses sign twice a cycle, which
    Q specifically penalizes), so this attenuates any real ~11.7Hz content
    in the transform at *every* q_ratio tested, from heavy (q_ratio=0.01,
    ~99.9% removed) to very loose (q_ratio=200, still ~3% removed) - there
    is no q_ratio that both meaningfully smooths per-frame noise and leaves
    genuine resonance-frequency motion untouched. The RTS backward pass also
    makes this *phase-flat* (~0deg shift regardless of q_ratio) even while
    strongly attenuating amplitude - so phase_lag_deg alone cannot detect
    this failure mode, only amplitude_ratio can. Treat Kalman smoothing as
    higher-risk than EMA for that reason, not lower-risk for being "more
    principled" - always check amplitude_ratio_11.7hz after using it.

    Args:
        z: (T,) raw measurements for one parameter
        r_scale: (T,) per-frame measurement-noise multiplier (see
            _measurement_noise_scale) - base R is scaled by this per frame
        q_ratio: process noise, as a fraction of the measurement sequence's
            own frame-to-frame variance - smaller means heavier smoothing
        dt: time step (frame units)

    Returns:
        (T,) smoothed parameter sequence
    """
    T = len(z)
    diffs = np.diff(z)
    base_var = float(np.var(diffs)) if T > 1 else 1.0
    base_var = max(base_var, 1e-12)
    r_base = base_var  # measurement noise on the same scale as the signal's own jitter
    q = q_ratio * base_var

    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    H = np.array([[1.0, 0.0]])

    x = np.array([z[0], 0.0])
    P = np.eye(2) * r_base

    x_filt = np.zeros((T, 2))
    P_filt = np.zeros((T, 2, 2))
    x_pred = np.zeros((T, 2))
    P_pred = np.zeros((T, 2, 2))
    x_filt[0], P_filt[0] = x, P
    x_pred[0], P_pred[0] = x, P

    for t in range(1, T):
        x_p = F @ x_filt[t - 1]
        P_p = F @ P_filt[t - 1] @ F.T + Q
        x_pred[t], P_pred[t] = x_p, P_p

        R = r_base * r_scale[t]
        S = (H @ P_p @ H.T).item() + R
        K = (P_p @ H.T) / S
        y = z[t] - (H @ x_p).item()
        x_f = x_p + (K.flatten() * y)
        P_f = (np.eye(2) - K @ H) @ P_p
        x_filt[t], P_filt[t] = x_f, P_f

    # RTS backward smoothing pass
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()
    for t in range(T - 2, -1, -1):
        C = P_filt[t] @ F.T @ np.linalg.pinv(P_pred[t + 1])
        x_smooth[t] = x_filt[t] + C @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + C @ (P_smooth[t + 1] - P_pred[t + 1]) @ C.T

    return x_smooth[:, 0]


def kalman_smooth(params, n_inliers, flow_err, q_ratio=1.0):
    """kalman_smooth_1d applied independently per parameter column."""
    r_scale = _measurement_noise_scale(n_inliers, flow_err)
    smoothed = np.empty_like(params)
    for j in range(params.shape[1]):
        smoothed[:, j] = kalman_smooth_1d(params[:, j], r_scale, q_ratio=q_ratio)
    return smoothed


class AffineFlowCompensator:
    VALID_MOTION_MODELS = ('affine', 'full_affine', 'homography')
    VALID_TRANSFORM_SMOOTHING = ('none', 'ema', 'kalman')

    def __init__(self, max_corners=250, quality_level=0.01, min_distance=10,
                 block_size=7, ransac_thresh=3.0, err_keep_pct=80.0,
                 lk_win_size=31, lk_max_level=4, motion_model='affine',
                 transform_smoothing='none', smoothing_alpha=0.3):
        if motion_model not in self.VALID_MOTION_MODELS:
            raise ValueError(f"motion_model must be one of {self.VALID_MOTION_MODELS}, got {motion_model!r}")
        if transform_smoothing not in self.VALID_TRANSFORM_SMOOTHING:
            raise ValueError(f"transform_smoothing must be one of {self.VALID_TRANSFORM_SMOOTHING}, "
                              f"got {transform_smoothing!r}")
        if transform_smoothing != 'none' and motion_model == 'homography':
            raise ValueError(
                "transform_smoothing is not supported with motion_model='homography': the "
                "translation/rotation/scale/shear decomposition it smooths doesn't represent "
                "homography's row-3 projective terms (h31, h32), so smoothing would silently "
                "discard them. Use motion_model in {'affine', 'full_affine'}."
            )
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.ransac_thresh = ransac_thresh
        self.err_keep_pct = err_keep_pct
        self.lk_win_size = lk_win_size
        self.lk_max_level = lk_max_level
        self.motion_model = motion_model
        self.min_points = 4 if motion_model == 'homography' else 8
        self.transform_smoothing = transform_smoothing
        self.smoothing_alpha = smoothing_alpha

        self.transform_list = None      # list of (3,3) homogeneous matrices actually used by
                                         # compensate_point - raw RANSAC fits if transform_smoothing
                                         # is 'none', else the smoothed version
        self.transform_list_raw = None  # the independent per-frame RANSAC fits, always unsmoothed
        self.transform_params_raw = None       # (T, 6) decomposed raw params, for debug logging
        self.transform_params_smoothed = None  # (T, 6) decomposed params actually used
        self.n_inliers = None         # (T,) diagnostic
        self.flow_err = None          # (T,) diagnostic

    def _apply_transform_smoothing(self):
        """Called once at the end of fit()/fit_streaming(), after
        self.transform_list (raw per-frame RANSAC fits) and the n_inliers/
        flow_err diagnostics are populated. Decomposes the raw sequence,
        optionally smooths it, and sets self.transform_list to whichever the
        rest of the pipeline (compensate_point) should actually use."""
        self.transform_list_raw = self.transform_list
        params_raw = decompose_transform_sequence(self.transform_list_raw)
        self.transform_params_raw = params_raw

        if self.transform_smoothing == 'none':
            self.transform_params_smoothed = params_raw
            return
        elif self.transform_smoothing == 'ema':
            params_smoothed = ema_smooth(params_raw, self.smoothing_alpha)
        elif self.transform_smoothing == 'kalman':
            params_smoothed = kalman_smooth(params_raw, self.n_inliers, self.flow_err)
        else:
            raise AssertionError(self.transform_smoothing)  # __init__ already validated

        self.transform_params_smoothed = params_smoothed
        self.transform_list = compose_transform_sequence(params_smoothed)

    def get_transform_debug_arrays(self):
        """Per-frame raw vs smoothed transform params, plus the diagnostics
        that drove Kalman's measurement-noise weighting - for checking (by
        hand, e.g. plotting) that smoothing isn't lagging behind real UAV
        motion at the VIV resonance, only removing per-frame RANSAC noise.
        Returns a dict of equal-length 1D arrays; None if fit()/
        fit_streaming() hasn't run yet."""
        if self.transform_params_raw is None:
            return None
        out = {'frame': np.arange(len(self.transform_params_raw)),
               'n_inliers': self.n_inliers, 'flow_err': self.flow_err}
        for j, name in enumerate(PARAM_NAMES):
            out[f'raw_{name}'] = self.transform_params_raw[:, j]
            out[f'smoothed_{name}'] = self.transform_params_smoothed[:, j]
        return out

    def _estimate_transform(self, p0, p1, err=None):
        """
        Shared RANSAC-fit step used by both fit() (small per-ROI patches) and
        fit_streaming() (full frames): optionally drop the worst err_keep_pct
        of points by LK error, then robustly fit the transform. Returns
        (M3 (3,3) or None, n_inliers, err_mean).
        """
        err_mean = np.nan
        if err is not None and len(err) == len(p0):
            e = np.asarray(err)
            if len(e) > 20:
                thr = np.nanpercentile(e, self.err_keep_pct)
                keep = e <= thr
                p0, p1, e = p0[keep], p1[keep], e[keep]
            err_mean = float(np.nanmean(e)) if len(e) else np.nan

        if len(p0) >= self.min_points:
            if self.motion_model == 'homography':
                M3, inliers = cv2.findHomography(
                    p0.astype(np.float32), p1.astype(np.float32),
                    method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh,
                    maxIters=2000, confidence=0.99,
                )
            elif self.motion_model == 'full_affine':
                # 6 DOF: independent x/y scale + shear, not just uniform
                # scale+rotation. Not used by the template-match + affine-flow configuration
                # (included for comparison only - see registries.py).
                M2, inliers = cv2.estimateAffine2D(
                    p0.astype(np.float32), p1.astype(np.float32),
                    method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh,
                    maxIters=2000, confidence=0.99,
                )
                M3 = None if M2 is None else np.vstack([M2, [0, 0, 1]])
            else:
                M2, inliers = cv2.estimateAffinePartial2D(
                    p0.astype(np.float32), p1.astype(np.float32),
                    method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh,
                    maxIters=2000, confidence=0.99,
                )
                M3 = None if M2 is None else np.vstack([M2, [0, 0, 1]])
            if M3 is not None:
                n_in = int(inliers.sum()) if inliers is not None else len(p0)
                return M3.astype(np.float64), n_in, err_mean

        if len(p0) >= 1:
            d = np.median(p1 - p0, axis=0)
            M3 = np.array([[1, 0, d[0]], [0, 1, d[1]], [0, 0, 1]], dtype=np.float64)
            return M3, len(p0), err_mean

        return None, 0, np.nan

    def fit_streaming(self, video_path, bg_boxes, max_frames=None):
        """
        Like fit(), but detects features and runs optical flow on FULL frames
        - streamed one at a time, holding only frame 0 (kept for the whole
        run) plus the current frame (discarded immediately after) - instead
        of small per-ROI patches.

        This is the fix for a real gap identified by reproducing a reference
        implementation exactly: cv2.calcOpticalFlowPyrLK with maxLevel=4 on a
        small (e.g. 67-100px) crop gives a degenerate pyramid (bottoms out
        around 4x4px); on a full 3840x2160 frame the same call gets a
        genuinely useful 4-level pyramid (3840->1920->960->480->240px) and
        (measured) a near-100% RANSAC inlier rate instead of ~85-90%.

        Args:
            video_path: path to video file
            bg_boxes: list of (x, y, w, h) background ROI bounding boxes, in
                full-frame pixel coordinates
            max_frames: optional cap on number of frames processed
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        ret, first_bgr = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Could not read first frame")
        first_gray = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY)

        pts0_list = []
        for (x, y, w, h) in bg_boxes:
            x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
            roi_img = first_gray[y:y + h, x:x + w]
            corners = cv2.goodFeaturesToTrack(
                roi_img, maxCorners=self.max_corners, qualityLevel=self.quality_level,
                minDistance=self.min_distance, blockSize=self.block_size, useHarrisDetector=False,
            )
            if corners is None:
                continue
            corners = corners.reshape(-1, 2)
            corners[:, 0] += x
            corners[:, 1] += y
            pts0_list.append(corners)

        if not pts0_list:
            cap.release()
            raise RuntimeError("AffineFlowCompensator.fit_streaming: no corner features "
                                "detected in any background ROI.")
        pts0 = np.vstack(pts0_list).astype(np.float32)
        n_total = len(pts0)

        transform_list = [np.eye(3, dtype=np.float64)]
        n_inliers = [n_total]
        flow_errs = [0.0]

        t = 1
        while True:
            if max_frames is not None and t >= max_frames:
                break
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            pts1, st, err = cv2.calcOpticalFlowPyrLK(
                first_gray, frame_gray, pts0.reshape(-1, 1, 2), None,
                winSize=(self.lk_win_size, self.lk_win_size), maxLevel=self.lk_max_level,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if pts1 is None or st is None:
                transform_list.append(transform_list[-1])
                n_inliers.append(0)
                flow_errs.append(np.nan)
                t += 1
                continue

            st_mask = st.reshape(-1).astype(bool)
            p0 = pts0[st_mask]
            p1 = pts1.reshape(-1, 2)[st_mask]
            e = err.reshape(-1)[st_mask] if err is not None else None

            if len(p0) == 0:
                transform_list.append(transform_list[-1])
                n_inliers.append(0)
                flow_errs.append(np.nan)
            else:
                M3, n_in, err_mean = self._estimate_transform(p0, p1, e)
                transform_list.append(M3 if M3 is not None else transform_list[-1])
                n_inliers.append(n_in)
                flow_errs.append(err_mean)
            t += 1

        cap.release()
        self.transform_list = transform_list
        self.n_inliers = np.array(n_inliers)
        self.flow_err = np.array(flow_errs)
        self._apply_transform_smoothing()

    def fit(self, bg_patches, bg_origins):
        """
        Args:
            bg_patches: list of (T, H, W) uint8 arrays, one per background ROI
            bg_origins: list of (x0, y0) - each patch's top-left corner in
                full-frame pixel coordinates, so features from different ROIs
                can be pooled into one consistent coordinate system
        """
        n_rois = len(bg_patches)
        T = bg_patches[0].shape[0]

        # Detect corners in frame 0 of each ROI, offset into pooled coords.
        pts0_per_roi = []
        for patch_stack, (ox, oy) in zip(bg_patches, bg_origins):
            frame0 = patch_stack[0]
            corners = cv2.goodFeaturesToTrack(
                frame0, maxCorners=self.max_corners, qualityLevel=self.quality_level,
                minDistance=self.min_distance, blockSize=self.block_size, useHarrisDetector=False,
            )
            if corners is None:
                pts0_per_roi.append(np.zeros((0, 2), dtype=np.float32))
                continue
            corners = corners.reshape(-1, 2)
            corners[:, 0] += ox
            corners[:, 1] += oy
            pts0_per_roi.append(corners)

        n_total = sum(len(p) for p in pts0_per_roi)
        if n_total == 0:
            raise RuntimeError("AffineFlowCompensator: no corner features detected in any "
                                "background ROI - pick larger or more textured background regions.")

        transform_list = [np.eye(3, dtype=np.float64)]  # frame 0: identity
        n_inliers = [n_total]
        flow_errs = [0.0]

        for t in range(1, T):
            p0_all, p1_all, err_all = [], [], []

            for patch_stack, (ox, oy), pts0_roi in zip(bg_patches, bg_origins, pts0_per_roi):
                if len(pts0_roi) == 0:
                    continue
                pts0_local = (pts0_roi - np.array([ox, oy])).astype(np.float32).reshape(-1, 1, 2)
                pts1_local, st, err = cv2.calcOpticalFlowPyrLK(
                    patch_stack[0], patch_stack[t], pts0_local, None,
                    winSize=(self.lk_win_size, self.lk_win_size), maxLevel=self.lk_max_level,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )
                if pts1_local is None or st is None:
                    continue
                st = st.reshape(-1).astype(bool)
                if not st.any():
                    continue
                pts1_roi = pts1_local.reshape(-1, 2)[st] + np.array([ox, oy])
                p0_all.append(pts0_roi[st])
                p1_all.append(pts1_roi)
                if err is not None:
                    flat_err = err.reshape(-1)[st]
                    if flat_err.size == len(pts1_roi):
                        err_all.append(flat_err)

            if not p0_all:
                transform_list.append(transform_list[-1])
                n_inliers.append(0)
                flow_errs.append(np.nan)
                continue

            p0 = np.concatenate(p0_all, axis=0)
            p1 = np.concatenate(p1_all, axis=0)
            e = None
            if err_all and sum(len(x) for x in err_all) == len(p0):
                e = np.concatenate(err_all, axis=0)

            M3, n_in, err_mean = self._estimate_transform(p0, p1, e)
            transform_list.append(M3 if M3 is not None else transform_list[-1])
            n_inliers.append(n_in)
            flow_errs.append(err_mean)

        self.transform_list = transform_list
        self.n_inliers = np.array(n_inliers)
        self.flow_err = np.array(flow_errs)
        self._apply_transform_smoothing()

    def compensate_point(self, target_traj_fullframe):
        """
        Args:
            target_traj_fullframe: (T, 2) target trajectory in full-frame
                pixel coordinates (NOT local ROI-patch coordinates)

        Returns:
            (T, 2) stabilized trajectory - target position with estimated
            camera affine motion undone
        """
        if self.transform_list is None:
            raise RuntimeError("Must call fit() first")
        T = target_traj_fullframe.shape[0]
        out = np.zeros_like(target_traj_fullframe, dtype=np.float64)
        for t in range(T):
            M_inv = np.linalg.inv(self.transform_list[t])
            x, y = target_traj_fullframe[t]
            p_h = M_inv @ np.array([x, y, 1.0])
            out[t, 0] = p_h[0] / p_h[2]
            out[t, 1] = p_h[1] / p_h[2]
        return out

    def get_diagnostics(self):
        """Per-frame RANSAC inlier counts and mean LK error, plus their mean -
                used to sanity-check that compensation actually had features to work
                with on every frame."""
        return {
            'n_inliers': self.n_inliers,
            'flow_err': self.flow_err,
            'mean_inliers': float(np.mean(self.n_inliers)) if self.n_inliers is not None else 0,
        }
