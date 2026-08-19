"""
Temporary helper: bootstrap a rois.json using the existing automated detectors
(structure-tensor saddle points + kinematic rigidity sort + PCA cable axis +
ray-cast calibration), instead of clicking manually.

This exists purely to unblock end-to-end testing of the tracking pipeline
without waiting on manual ROI selection. It writes the exact same JSON schema
as data/roi_select.py, so it is a drop-in alternative, not a dependency of
the tracking code. Prefer data/roi_select.py once you want to sanity-check
or override the automatically detected points.

Usage:
    python data/bootstrap_rois_auto.py <video_path> [--out configs/rois.json]
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.video import load_video_frames, detect_cable_axis_automated, calibrate_scale_ray_cast
from data.query_points import detect_all_saddle_points, sort_by_kinematic_rigidity
from data.roi_io import save_rois, viv_direction_from_axis_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path')
    parser.add_argument('--out', default=None)
    parser.add_argument('--checkerboard-mm', type=float, default=55.0)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = args.out or os.path.join(project_root, 'configs', 'rois.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[INFO] Loading first 50 frames (every 10th) from {args.video_path}...")
    cap = cv2.VideoCapture(args.video_path)
    ret, first_frame_bgr = cap.read()
    if not ret:
        raise ValueError("Could not read first frame.")
    first_frame_gray = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2GRAY)

    pre_flight_frames = [first_frame_gray]
    frame_idx = 0
    while len(pre_flight_frames) < 50:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 10 == 0:
            pre_flight_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    print("[INFO] Detecting saddle points...")
    raw_saddle_points = detect_all_saddle_points(first_frame_gray, expected_points=3)

    print("[INFO] Sorting by kinematic rigidity (target vs. rigid background)...")
    target_pt, bg_pts = sort_by_kinematic_rigidity(pre_flight_frames, raw_saddle_points)

    print("[INFO] Calibrating mm/px via ray-cast gradient edges...")
    mm_per_px = calibrate_scale_ray_cast(first_frame_gray, target_pt, checkerboard_size_mm=args.checkerboard_mm)

    print("[INFO] Detecting cable axis (PCA on red mask)...")
    u_normal, u_axis = detect_cable_axis_automated(first_frame_bgr)
    # Reconstruct two axis points from the unit vector, anchored at target_pt,
    # so downstream storage/schema stays identical to the manual-click path.
    axis_point_a = target_pt
    axis_point_b = target_pt + u_axis * 200.0
    _, projection_angle_rad = viv_direction_from_axis_points(axis_point_a, axis_point_b)

    data = save_rois(
        out_path,
        video_path=args.video_path,
        target_point=target_pt,
        bg_points=bg_pts,
        mm_per_px=mm_per_px,
        calibration_method='auto_ray_cast',
        checkerboard_size_mm=args.checkerboard_mm,
        cable_point_a=axis_point_a,
        cable_point_b=axis_point_b,
        viv_direction=u_normal,
        projection_angle_rad=projection_angle_rad,
    )

    print("\n" + "=" * 70)
    print(f"Saved to {out_path}")
    print(f"  target_point   = {data['target_point']}")
    print(f"  bg_points      = {data['bg_points']}")
    print(f"  mm_per_px      = {data['calibration']['mm_per_px']:.5f}")
    print(f"  viv_direction  = {data['cable_axis']['viv_direction']}")
    print("=" * 70)


if __name__ == '__main__':
    main()
