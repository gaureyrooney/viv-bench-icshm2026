"""
ROI/calibration JSON schema: load & save.

This is the single hand-off point between "where is the target" (segmentation /
ROI selection - manual or automated, doesn't matter) and "how does it move"
(tracking). Anything upstream just needs to produce a dict matching this shape;
anything downstream (pipeline.py) only ever reads it back via load_rois().

Schema:
{
  "video_path": str,
  "reference_frame_index": int,
  "target_point": [x, y],
  "bg_points": [[x, y], ...],
  "bg_boxes": [[x, y, w, h], ...] | null,   # exact bounding boxes, if known -
                                             # needed for full-frame-pyramid
                                             # optical flow compensation
                                             # (AffineFlowCompensator.fit_streaming);
                                             # falls back to a symmetric box
                                             # around bg_points when absent.
  "calibration": {
    "method": "manual_edge" | "checkerboard_bbox" | ...,
    "checkerboard_size_mm": float,
    "mm_per_px": float
  },
  "cable_axis": {
    "point_a": [x, y],
    "point_b": [x, y],
    "viv_direction": [nx, ny],       # unit normal, perpendicular to cable
    "projection_angle_rad": float
  },
  "patch_size": int
}
"""
import json
import numpy as np


def save_rois(path, video_path, target_point, bg_points, mm_per_px,
              calibration_method, checkerboard_size_mm,
              cable_point_a, cable_point_b, viv_direction, projection_angle_rad,
              reference_frame_index=0, patch_size=100, bg_boxes=None):
    data = {
        'video_path': str(video_path),
        'reference_frame_index': int(reference_frame_index),
        'target_point': [float(target_point[0]), float(target_point[1])],
        'bg_points': [[float(p[0]), float(p[1])] for p in bg_points],
        'bg_boxes': None if bg_boxes is None else [
            [float(x), float(y), float(w), float(h)] for (x, y, w, h) in bg_boxes
        ],
        'calibration': {
            'method': calibration_method,
            'checkerboard_size_mm': float(checkerboard_size_mm),
            'mm_per_px': float(mm_per_px),
        },
        'cable_axis': {
            'point_a': [float(cable_point_a[0]), float(cable_point_a[1])],
            'point_b': [float(cable_point_b[0]), float(cable_point_b[1])],
            'viv_direction': [float(viv_direction[0]), float(viv_direction[1])],
            'projection_angle_rad': float(projection_angle_rad),
        },
        'patch_size': int(patch_size),
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return data


def load_rois(path):
    """Read and validate an ROI definition JSON, raising on a missing required field."""
    with open(path, 'r') as f:
        data = json.load(f)

    data['target_point'] = np.array(data['target_point'], dtype=np.float64)
    data['bg_points'] = [np.array(p, dtype=np.float64) for p in data['bg_points']]
    data.setdefault('bg_boxes', None)
    data['cable_axis']['viv_direction'] = np.array(
        data['cable_axis']['viv_direction'], dtype=np.float64
    )
    return data


def viv_direction_from_axis_points(point_a, point_b):
    """
    Given two points along the cable, return (unit_normal, projection_angle_rad).
    Normal points "up" in image space (negative Y) by convention, matching
    detect_cable_axis_automated in data/video.py.
    """
    point_a = np.asarray(point_a, dtype=np.float64)
    point_b = np.asarray(point_b, dtype=np.float64)

    u_axis = point_b - point_a
    norm = np.linalg.norm(u_axis)
    if norm < 1e-6:
        raise ValueError("Cable axis points are coincident.")
    u_axis = u_axis / norm

    if u_axis[0] < 0:
        u_axis = -u_axis

    u_normal = np.array([-u_axis[1], u_axis[0]], dtype=np.float64)
    u_normal = u_normal / np.linalg.norm(u_normal)
    if u_normal[1] > 0:
        u_normal = -u_normal

    projection_angle_rad = abs(np.arctan2(u_axis[1], u_axis[0]))
    return u_normal, projection_angle_rad
