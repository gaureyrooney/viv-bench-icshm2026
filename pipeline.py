"""
ROI-driven VIV tracking pipeline.

Deliberately decoupled from ROI selection: this module only ever reads a
rois.json (see data/roi_io.py) describing where the target and background
reference points are. That JSON can come from manual clicking
(data/roi_select.py) or automated detection (data/bootstrap_rois_auto.py) -
the pipeline doesn't know or care which. Swapping ROI source never requires
touching this file.

Stages:
  1. Stream-decode the video, cropping small patches around each query point
     as we go (never materializes full-resolution 4K frames).
  2. Track the target patch + every background patch with a pluggable Tracker.
  3. Compensate camera ego-motion using a pluggable Compensator, driven by the
     mean motion of ALL background points (not just one).
  4. Project 2D pixel displacement onto the cable-perpendicular (VIV) axis.
  5. Convert to mm using the calibrated scale.
  6. Optionally align + compare against an LDS reference signal.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.video import load_roi_patches
from data.roi_io import load_rois
from dsp.alignment import align_signals, mean_subtract


class ROITracker:
    """
    End-to-end tracking pipeline driven by a pre-computed rois.json.
    """

    def __init__(self, video_path, rois_json_path, patch_size=None):
        self.video_path = video_path
        self.roi_data = load_rois(rois_json_path)

        self.target_point = self.roi_data['target_point']
        self.bg_points = self.roi_data['bg_points']
        self.bg_boxes = self.roi_data.get('bg_boxes')
        self.mm_per_px = self.roi_data['calibration']['mm_per_px']
        self.viv_direction = self.roi_data['cable_axis']['viv_direction']
        self.patch_size = patch_size or self.roi_data.get('patch_size', 100)

        print(f"[INFO] Loaded ROIs from {rois_json_path}")
        print(f"    -> Target: {self.target_point}")
        print(f"    -> Background refs: {len(self.bg_points)}")
        print(f"    -> Scale: {self.mm_per_px:.5f} mm/px")
        print(f"    -> VIV direction: {self.viv_direction}")

    def _query_points_dict(self):
        points = {'target': tuple(self.target_point)}
        for i, p in enumerate(self.bg_points):
            points[f'bg_{i}'] = tuple(p)
        return points

    def track_rois(self, tracker, max_frames=None):
        """Track the target and every background point with `tracker`.

                Decodes the video once, cropping only the ROI patches (see
                data/video.py), then runs the tracker on each patch stack.

                Returns a tracking_dict with target/background trajectories in
                patch-local pixel coordinates, plus tracker and video metadata.
                """
        points = self._query_points_dict()
        print(f"\n[1] Streaming video, cropping {len(points)} ROI(s) "
              f"({self.patch_size}x{self.patch_size} px each)...")
        patches, meta_video = load_roi_patches(
            self.video_path, points, patch_size=self.patch_size, max_frames=max_frames
        )
        print(f"    -> {meta_video['frame_count']} frames streamed "
              f"({meta_video['duration_s']:.2f} s @ {meta_video['fps']:.1f} fps)")

        local_query = np.array([[self.patch_size / 2, self.patch_size / 2]], dtype=np.float32)

        print(f"\n[2] Tracking target with {tracker.name}...")
        traj_target, conf_target, meta_target = tracker.track(patches['target'], local_query)

        bg_trajs = []
        for i in range(len(self.bg_points)):
            print(f"[2] Tracking background reference {i} with {tracker.name}...")
            traj_bg, _, _ = tracker.track(patches[f'bg_{i}'], local_query)
            bg_trajs.append(traj_bg)

        return {
            'target_trajectory_px': traj_target,
            'bg_trajectories_px': bg_trajs,
            'target_confidence': conf_target,
            'tracker_meta': meta_target,
            'video_meta': meta_video,
        }

    def track_target_and_bg_patches(self, tracker, max_frames=None):
        """
        Like track_rois, but keeps the raw background ROI patches (+ their
        full-frame origins) instead of collapsing each one to a single
        tracked point - needed by AffineFlowCompensator, which detects and
        optical-flow-tracks many corner features per background ROI itself.
        """
        points = self._query_points_dict()
        print(f"\n[1] Streaming video, cropping {len(points)} ROI(s) "
              f"({self.patch_size}x{self.patch_size} px each)...")
        patches, meta_video = load_roi_patches(
            self.video_path, points, patch_size=self.patch_size, max_frames=max_frames
        )
        print(f"    -> {meta_video['frame_count']} frames streamed "
              f"({meta_video['duration_s']:.2f} s @ {meta_video['fps']:.1f} fps)")

        local_query = np.array([[self.patch_size / 2, self.patch_size / 2]], dtype=np.float32)

        print(f"\n[2] Tracking target with {tracker.name}...")
        traj_target, conf_target, meta_target = tracker.track(patches['target'], local_query)

        bg_patches = [patches[f'bg_{i}'] for i in range(len(self.bg_points))]
        bg_origins = [meta_video['origins'][f'bg_{i}'] for i in range(len(self.bg_points))]
        target_origin = meta_video['origins']['target']

        return {
            'target_trajectory_px': traj_target,
            'target_confidence': conf_target,
            'tracker_meta': meta_target,
            'target_origin': target_origin,
            'bg_patches': bg_patches,
            'bg_origins': bg_origins,
            'video_meta': meta_video,
        }

    def track_target_only(self, tracker, max_frames=None):
        """
        Like track_target_and_bg_patches, but skips decoding/tracking
        background patches entirely - for when ego-motion compensation gets
        its background data a different way (e.g. AffineFlowCompensator.
        fit_streaming, which does its own full-frame video pass).
        """
        points = {'target': tuple(self.target_point)}
        print(f"\n[1] Streaming video, cropping target ROI ({self.patch_size}x{self.patch_size} px)...")
        patches, meta_video = load_roi_patches(
            self.video_path, points, patch_size=self.patch_size, max_frames=max_frames
        )
        print(f"    -> {meta_video['frame_count']} frames streamed "
              f"({meta_video['duration_s']:.2f} s @ {meta_video['fps']:.1f} fps)")

        local_query = np.array([[self.patch_size / 2, self.patch_size / 2]], dtype=np.float32)
        print(f"\n[2] Tracking target with {tracker.name}...")
        traj_target, conf_target, meta_target = tracker.track(patches['target'], local_query)

        return {
            'target_trajectory_px': traj_target,
            'target_confidence': conf_target,
            'tracker_meta': meta_target,
            'target_origin': meta_video['origins']['target'],
            'video_meta': meta_video,
        }

    def run_full_pipeline_affine_flow(self, tracker, affine_compensator, lds_ref=None, max_frames=None):
        """
        Same pipeline as run_full_pipeline, but ego-motion compensation comes
        from AffineFlowCompensator (many background corner features, optical
        flow, RANSAC-fit affine per frame) instead of a point-trajectory
        Compensator. Everything downstream (projection, mm conversion,
        alignment) is identical - only how camera motion gets removed differs.

        Uses AffineFlowCompensator.fit_streaming (full video frames, matching
        a verified template-match + affine-flow configuration's optical-flow pyramid quality)
        when self.bg_boxes is available from the ROI source; falls back to
        the small-patch fit() otherwise.
        """
        print("\n" + "=" * 70)
        print("EXECUTING VIV TRACKING PIPELINE (affine-flow compensation)")
        print("=" * 70)

        use_streaming = self.bg_boxes is not None and hasattr(affine_compensator, 'fit_streaming')

        if use_streaming:
            td = self.track_target_only(tracker, max_frames=max_frames)
            print(f"\n[3] Fitting affine ego-motion model from {len(self.bg_boxes)} "
                  f"background ROI(s) (full-frame optical flow)...")
            affine_compensator.fit_streaming(self.video_path, self.bg_boxes, max_frames=max_frames)
        else:
            td = self.track_target_and_bg_patches(tracker, max_frames=max_frames)
            print(f"\n[3] Fitting affine ego-motion model from {len(td['bg_patches'])} "
                  f"background ROI(s) (small-patch optical flow)...")
            affine_compensator.fit(td['bg_patches'], td['bg_origins'])

        diag = affine_compensator.get_diagnostics()
        print(f"    -> Mean RANSAC inliers/frame: {diag['mean_inliers']:.1f}")

        # Local (in-patch) target trajectory -> full-frame pixel coordinates,
        # since the affine model was fit in full-frame coordinates.
        T = td['target_trajectory_px'].shape[0]
        traj_target_local = td['target_trajectory_px'].reshape(T, 2)
        ox, oy = td['target_origin']
        traj_target_full = traj_target_local + np.array([ox, oy])

        traj_stabilized = affine_compensator.compensate_point(traj_target_full)
        # Displacement relative to the initial (frame-0) stabilized position.
        traj_disp = traj_stabilized - traj_stabilized[0]

        print("\n[4] Projecting to VIV axis...")
        disp_px_1d = np.dot(traj_disp, self.viv_direction)

        print("\n[5] Converting to millimeters...")
        disp_mm_1d = disp_px_1d * self.mm_per_px

        if lds_ref is not None:
            print("\n[6] Cross-correlating with LDS ground truth...")
            disp_mm_aligned, lds_aligned, lag_s, _, align_sign = align_signals(
                disp_mm_1d, lds_ref[:len(disp_mm_1d)], fs=50, max_lag_s=1.0
            )
        else:
            disp_mm_aligned = mean_subtract(disp_mm_1d)
            lds_aligned = None
            lag_s = None

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f" -> Measured VIV range: [{disp_mm_aligned.min():.4f}, {disp_mm_aligned.max():.4f}] mm")
        if lag_s is not None:
            rmse_val = np.sqrt(np.mean((disp_mm_aligned - lds_aligned) ** 2))
            print(f" -> Temporal alignment lag: {lag_s:.3f} s (polarity {align_sign:+.0f})")
            print(f" -> RMSE vs LDS: {rmse_val:.4f} mm")

        return {
            'displacement_mm': disp_mm_aligned,
            'displacement_px': disp_px_1d,
            'lds_reference': lds_aligned,
            'lag_s': lag_s,
            'mm_per_px': self.mm_per_px,
            'tracking_dict': td,
            'affine_diagnostics': diag,
            'video_meta': td['video_meta'],
        }

    def apply_compensation(self, tracking_dict, compensator):
        """Remove estimated camera motion from the target trajectory.

                Takes a point-trajectory compensator (differential / none). The
                affine_flow compensator needs raw pixels instead and goes through
                run_full_pipeline_affine_flow.
                """
        traj_target = tracking_dict['target_trajectory_px']
        T = traj_target.shape[0]
        traj_target_2d = traj_target.reshape(T, -1, 2)

        bg_trajs = tracking_dict['bg_trajectories_px']
        if len(bg_trajs) > 0:
            # (T, n_bg, 2): every background point's trajectory, so the
            # compensator averages across all rigid references instead of
            # relying on a single (noisier) one.
            bg_stack = np.concatenate(
                [b.reshape(T, -1, 2) for b in bg_trajs], axis=1
            )
        else:
            bg_stack = np.zeros((T, 0, 2), dtype=np.float32)

        compensator.fit(traj_target_2d, bg_stack)
        traj_comp = compensator.compensate(traj_target_2d)

        bg_motion = compensator.get_background_motion()
        print(f"\n[3] Applied {compensator.__class__.__name__}.")
        if bg_motion is not None:
            # bg_motion is the mean *absolute position* trajectory of the
            # background points (in local ROI-patch coordinates, centered
            # near patch_size/2, not zero) - std (fluctuation around its own
            # mean) is the meaningful "jitter" quantity, not raw RMS.
            bg_jitter_std = np.std(bg_motion, axis=0)
            print(f"    -> UAV jitter std (from {bg_stack.shape[1]} bg point(s)): {bg_jitter_std} px")

        return {'trajectory_px': traj_comp, 'bg_motion_px': bg_motion}

    def project_to_viv_direction(self, trajectory_px):
        """Project a 2-D trajectory onto the cable-axis normal (the VIV direction),
                collapsing it to the scalar signal the LDS also measures."""
        if trajectory_px.ndim == 3:
            disp_per_point = np.dot(trajectory_px, self.viv_direction)
            return np.mean(disp_per_point, axis=1)
        return np.dot(trajectory_px, self.viv_direction)

    def run_full_pipeline(self, tracker, compensator, lds_ref=None, max_frames=None):
        print("\n" + "=" * 70)
        print("EXECUTING VIV TRACKING PIPELINE")
        print("=" * 70)

        tracking_dict = self.track_rois(tracker, max_frames=max_frames)
        return self.run_from_trajectories(tracking_dict, compensator, lds_ref=lds_ref)

    def run_from_trajectories(self, tracking_dict, compensator, lds_ref=None):
        """
        Same as run_full_pipeline from the compensation step onward, but
        starting from an already-computed tracking_dict (e.g. loaded from a
        cached trajectories.npz) instead of re-tracking. This is what makes
        comparing compensators/filters/projection choices cheap: tracking
        (the expensive step) runs once, everything downstream re-runs in
        under a second.
        """
        comp_dict = self.apply_compensation(tracking_dict, compensator)

        print("\n[4] Projecting to VIV axis...")
        disp_px_1d = self.project_to_viv_direction(comp_dict['trajectory_px'])

        print("\n[5] Converting to millimeters...")
        disp_mm_1d = disp_px_1d * self.mm_per_px

        if lds_ref is not None:
            print("\n[6] Cross-correlating with LDS ground truth...")
            disp_mm_aligned, lds_aligned, lag_s, _, align_sign = align_signals(
                disp_mm_1d, lds_ref[:len(disp_mm_1d)], fs=50, max_lag_s=1.0
            )
        else:
            disp_mm_aligned = mean_subtract(disp_mm_1d)
            lds_aligned = None
            lag_s = None

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f" -> Measured VIV range: [{disp_mm_aligned.min():.4f}, {disp_mm_aligned.max():.4f}] mm")
        if lag_s is not None:
            rmse_val = np.sqrt(np.mean((disp_mm_aligned - lds_aligned) ** 2))
            print(f" -> Temporal alignment lag: {lag_s:.3f} s (polarity {align_sign:+.0f})")
            print(f" -> RMSE vs LDS: {rmse_val:.4f} mm")

        return {
            'displacement_mm': disp_mm_aligned,
            'displacement_px': disp_px_1d,
            'lds_reference': lds_aligned,
            'lag_s': lag_s,
            'mm_per_px': self.mm_per_px,
            'tracking_dict': tracking_dict,
            'comp_dict': comp_dict,
            'video_meta': tracking_dict['video_meta'],
        }
