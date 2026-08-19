"""
Import a manually-extracted ROI JSON (the {target_roi, bg_rois, cable_axis}
bounding-box format used during earlier manual ROI selection, e.g.
"dense/data/rois.json") into this project's rois.json schema (data/roi_io.py).

This is an alternative front-door into the same decoupled ROI hand-off used by
data/roi_select.py and data/bootstrap_rois_auto.py - the tracking pipeline
doesn't know or care which of the three produced configs/rois.json.

Input format:
{
  "target_roi": [x, y, w, h],
  "bg_rois": [[x, y, w, h], ...],       # or "stab_rois" (the template-match + affine-flow configuration's key name)
  "cable_axis": [[x1, y1], [x2, y2]]    # or "cable_axis_p1"/"cable_axis_p2"
}

Usage:
    python data/import_manual_rois.py <manual_rois.json> [--out configs/rois.json] \
        [--checkerboard-mm 55.0] [--max-bg N] [--video <path>]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.roi_io import save_rois, viv_direction_from_axis_points


def roi_center(roi):
    x, y, w, h = roi
    return (x + w / 2.0, y + h / 2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manual_rois_json')
    parser.add_argument('--out', default=None)
    parser.add_argument('--checkerboard-mm', type=float, default=55.0)
    parser.add_argument('--max-bg', type=int, default=None,
                         help='Use only the first N bg_rois (e.g. to skip a large non-point '
                              'stabilization region meant for a different technique)')
    parser.add_argument('--video', default=None, help='Override video_path stored in the output JSON')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = args.out or os.path.join(project_root, 'configs', 'rois.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(args.manual_rois_json, 'r') as f:
        manual = json.load(f)

    target_point = roi_center(manual['target_roi'])
    tw, th = manual['target_roi'][2], manual['target_roi'][3]
    mm_per_px = manual.get('mm_per_px') or (args.checkerboard_mm / ((tw + th) / 2.0))

    bg_rois = manual.get('bg_rois', manual.get('stab_rois'))
    if bg_rois is None:
        raise KeyError("Input JSON must have 'bg_rois' or 'stab_rois'")
    n_bg_total = len(bg_rois)
    if args.max_bg is not None:
        bg_rois = bg_rois[:args.max_bg]
    bg_points = [roi_center(r) for r in bg_rois]
    bg_boxes = [tuple(r) for r in bg_rois]

    if 'cable_axis' in manual:
        axis_a, axis_b = manual['cable_axis']
    else:
        axis_a, axis_b = manual['cable_axis_p1'], manual['cable_axis_p2']
    viv_direction, projection_angle_rad = viv_direction_from_axis_points(axis_a, axis_b)

    data = save_rois(
        out_path,
        video_path=args.video or manual.get('video', ''),
        target_point=target_point,
        bg_points=bg_points,
        bg_boxes=bg_boxes,
        mm_per_px=mm_per_px,
        calibration_method='manual_target_bbox',
        checkerboard_size_mm=args.checkerboard_mm,
        cable_point_a=axis_a,
        cable_point_b=axis_b,
        viv_direction=viv_direction,
        projection_angle_rad=projection_angle_rad,
    )

    print(f"Imported {args.manual_rois_json} -> {out_path}")
    print(f"  target_point = {data['target_point']}  (bbox {manual['target_roi']})")
    print(f"  bg_points    = {data['bg_points']}  ({len(bg_points)}/{n_bg_total} used)")
    print(f"  bg_boxes     = {data['bg_boxes']}")
    print(f"  mm_per_px    = {data['calibration']['mm_per_px']:.6f}")
    print(f"  viv_direction= {data['cable_axis']['viv_direction']}")


if __name__ == '__main__':
    main()
