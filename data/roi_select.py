"""
Manual ROI selection tool (decoupled from tracking).

Run this once per video. Click points in the displayed window; it writes a
rois.json that pipeline.py can consume independently. No tracking, no
segmentation model - just point-and-click, so it's a completely separate
step from "how well does method X track".

Usage:
    python data/roi_select.py <video_path> [--out configs/rois.json] [--frame 0]

Steps, in order:
    1. Click the TARGET point (checkerboard center on the vibrating cable) - 1 click
    2. Click BACKGROUND reference points (rigid/stationary features) - 2+ clicks,
       press 'n' when done
    3. Click the two ends of ONE EDGE of the 55 mm checkerboard target (for
       mm/px calibration) - 2 clicks
    4. Click two points along the CABLE AXIS (for VIV projection direction) - 2 clicks

Keys:
    n       - finish current step (only needed for the background-points step)
    r       - restart the current step
    u       - undo last click in the current step
    ESC / q - quit without saving
    ENTER   - save once all steps are complete
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.roi_io import save_rois, viv_direction_from_axis_points

STEPS = [
    ('target', 1, 'Click the TARGET point (checkerboard on the cable)'),
    ('bg', -1, "Click background reference points, press 'n' when done (need >=1)"),
    ('calib', 2, 'Click the two ends of ONE EDGE of the checkerboard target (55 mm)'),
    ('axis', 2, 'Click two points along the CABLE AXIS'),
]

DISPLAY_MAX_W = 1600


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path')
    parser.add_argument('--out', default=None, help='Output JSON path (default: configs/rois.json next to this script)')
    parser.add_argument('--frame', type=int, default=0, help='Reference frame index to select on')
    parser.add_argument('--checkerboard-mm', type=float, default=55.0)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = args.out or os.path.join(project_root, 'configs', 'rois.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {args.video_path}")
    for _ in range(args.frame + 1):
        ret, frame_bgr = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {args.frame} from video.")
    cap.release()

    h, w = frame_bgr.shape[:2]
    scale = min(1.0, DISPLAY_MAX_W / w)
    disp_w, disp_h = int(w * scale), int(h * scale)
    base_display = cv2.resize(frame_bgr, (disp_w, disp_h))

    state = {'step_idx': 0, 'clicks': [[] for _ in STEPS]}

    def to_full_res(pt_disp):
        return (pt_disp[0] / scale, pt_disp[1] / scale)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            name, n_required, _ = STEPS[state['step_idx']]
            state['clicks'][state['step_idx']].append((x, y))
            if n_required > 0 and len(state['clicks'][state['step_idx']]) >= n_required:
                advance_step()

    def advance_step():
        if state['step_idx'] < len(STEPS) - 1:
            state['step_idx'] += 1

    win = 'ROI Selection - see terminal for instructions'
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    colors = {'target': (0, 0, 255), 'bg': (0, 255, 0), 'calib': (255, 0, 0), 'axis': (0, 255, 255)}

    print("=" * 70)
    print("ROI SELECTION")
    print("=" * 70)
    for name, n_required, instr in STEPS:
        print(f"  [{name}] {instr}")
    print("Keys: n=next step, r=restart step, u=undo click, ENTER=save, ESC/q=quit")
    print("=" * 70)

    last_step_printed = -1
    while True:
        if state['step_idx'] != last_step_printed:
            print(f"\n>> Step '{STEPS[state['step_idx']][0]}': {STEPS[state['step_idx']][2]}")
            last_step_printed = state['step_idx']

        canvas = base_display.copy()
        for name, _, _ in STEPS:
            idx = [s[0] for s in STEPS].index(name)
            for (x, y) in state['clicks'][idx]:
                cv2.circle(canvas, (x, y), 5, colors[name], -1)
        cv2.putText(canvas, f"Step: {STEPS[state['step_idx']][0]}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord('q')):
            print("Aborted, nothing saved.")
            cv2.destroyAllWindows()
            return
        elif key == ord('n'):
            advance_step()
        elif key == ord('r'):
            state['clicks'][state['step_idx']] = []
        elif key == ord('u'):
            if state['clicks'][state['step_idx']]:
                state['clicks'][state['step_idx']].pop()
        elif key in (13, 10):  # Enter
            if all(len(state['clicks'][i]) >= (STEPS[i][1] if STEPS[i][1] > 0 else 1) for i in range(len(STEPS))):
                break
            print("Not all steps have enough points yet.")

    cv2.destroyAllWindows()

    target_point = to_full_res(state['clicks'][0][0])
    bg_points = [to_full_res(p) for p in state['clicks'][1]]
    calib_pts = [to_full_res(p) for p in state['clicks'][2]]
    axis_pts = [to_full_res(p) for p in state['clicks'][3]]

    edge_px = float(np.linalg.norm(np.array(calib_pts[0]) - np.array(calib_pts[1])))
    mm_per_px = args.checkerboard_mm / edge_px

    viv_direction, projection_angle_rad = viv_direction_from_axis_points(axis_pts[0], axis_pts[1])

    data = save_rois(
        out_path,
        video_path=args.video_path,
        target_point=target_point,
        bg_points=bg_points,
        mm_per_px=mm_per_px,
        calibration_method='manual_edge',
        checkerboard_size_mm=args.checkerboard_mm,
        cable_point_a=axis_pts[0],
        cable_point_b=axis_pts[1],
        viv_direction=viv_direction,
        projection_angle_rad=projection_angle_rad,
        reference_frame_index=args.frame,
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
