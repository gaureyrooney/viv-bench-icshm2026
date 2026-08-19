"""
Video loader: Frame extraction, PCA axis detection, and Ray-Cast Scale Calibration.
(Note: Legacy OpenCV checkerboard functions have been deprecated in favor of 
Structure Tensor extraction in query_points.py)
"""
import numpy as np
import cv2
from pathlib import Path

def load_video_frames(video_path, max_frames=None, grayscale=False):
    """
    Decode all frames from video into a NumPy array.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"[ERROR] Cannot open video: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
        
    frames_list = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        frames_list.append(frame)
        frame_idx += 1
        
        if max_frames is not None and frame_idx >= max_frames:
            break
            
    cap.release()
    
    frames = np.array(frames_list, dtype=np.uint8)
    
    metadata = {
        'fps': fps,
        'frame_count': frames.shape[0],
        'height': frames.shape[1],
        'width': frames.shape[2],
        'duration_s': frames.shape[0] / fps,
    }
    
    return frames, metadata


def load_roi_patches(video_path, points, patch_size=100, sizes=None, max_frames=None):
    """
    Stream-decode a video once and crop fixed-size patches around each named
    point from every frame, without ever materializing full-resolution frames.

    A 4K (3840x2160) 60s@50fps clip is ~23 GB as a full uint8 grayscale stack -
    infeasible to hold in RAM. Since tracking only needs small ROI patches
    around a handful of query points, we crop during decode instead.

    Args:
        video_path: path to video file
        points: dict {name: (x, y)} of patch centers in frame coordinates
                (typically from the reference frame used for ROI selection)
        patch_size: default side length (px) of each square patch
        sizes: optional dict {name: (w, h)} overriding patch_size per point -
            e.g. to crop a background ROI to its exact bounding box instead of
            a uniform default, matching a reference selection precisely
            rather than an arbitrary fixed size.
        max_frames: optional cap on number of frames to read

    Returns:
        patches: dict {name: (T, h, w) uint8 array} (h=w=patch_size unless
            overridden via `sizes`)
        metadata: dict with fps, frame_count, duration_s, origins, sizes
    """
    sizes = sizes or {}
    point_sizes = {name: sizes.get(name, (patch_size, patch_size)) for name in points}
    origins = {
        name: (int(round(x - point_sizes[name][0] / 2)), int(round(y - point_sizes[name][1] / 2)))
        for name, (x, y) in points.items()
    }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"[ERROR] Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    buffers = {name: [] for name in points}
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        for name, (x0, y0) in origins.items():
            w, h = point_sizes[name]
            x1, y1 = x0 + w, y0 + h
            src_x0, src_y0 = max(0, x0), max(0, y0)
            src_x1, src_y1 = min(frame_w, x1), min(frame_h, y1)

            patch = np.zeros((h, w), dtype=np.uint8)
            if src_x1 > src_x0 and src_y1 > src_y0:
                patch[src_y0 - y0:src_y1 - y0, src_x0 - x0:src_x1 - x0] = \
                    frame_gray[src_y0:src_y1, src_x0:src_x1]
            buffers[name].append(patch)

        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()

    patches = {name: np.stack(buf, axis=0) for name, buf in buffers.items()}
    metadata = {
        'fps': fps,
        'frame_count': frame_idx,
        'duration_s': frame_idx / fps if fps else None,
        'patch_size': patch_size,
        'origins': origins,
        'sizes': point_sizes,
    }
    return patches, metadata


def detect_cable_axis_automated(frame_bgr):
    """
    Automatically detects the 2D normal vector of the red stay cable 
    in the camera's actual frame, accounting for UAV gimbal roll.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    
  # 2. Strict Red Mask (High saturation/value to ignore dark floor rust)
    lower_red_1 = np.array([0, 150, 100])
    upper_red_1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    
    lower_red_2 = np.array([170, 150, 100])
    upper_red_2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    
    red_mask = mask1 + mask2
    
    # 3. Geometric Mute: The floor is at the bottom. Erase the bottom 50% of the mask.
    height = red_mask.shape[0]
    red_mask[int(height * 0.5):, :] = 0
    
    y_coords, x_coords = np.nonzero(red_mask)
    if len(x_coords) < 100:
        raise ValueError("[ERROR] Failed to detect enough red pixels to form a cable axis.")
        
    # PCA to find the line of best fit
    points = np.vstack((x_coords, y_coords)).T
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points.astype(np.float32), mean=None)
    
    u_axis = eigenvectors[0]
    
    # Ensure the vector points "rightwards" for consistency
    if u_axis[0] < 0:
        u_axis = -u_axis
        
    # Calculate 2D Normal Vector (perpendicular VIV projection plane)
    u_normal = np.array([-u_axis[1], u_axis[0]], dtype=np.float64)
    u_normal = u_normal / np.linalg.norm(u_normal)
    
    # Ensure the normal vector points generally "up" (negative Y in image space)
    if u_normal[1] > 0:
        u_normal = -u_normal
        
    return u_normal, u_axis


def calibrate_scale_ray_cast(frame_gray, center_point, checkerboard_size_mm=55.0):
    """
    Dynamically calculates mm/px scale by isolating the sharp gradient 
    boundaries of the 2x2 target, completely immune to shadow/clutter area.
    """
    cx, cy = int(center_point[0]), int(center_point[1])
    
    # 1. Crop a tight patch around the mathematically perfect saddle point
    pad = 80 
    patch = frame_gray[max(0, cy-pad):cy+pad, max(0, cx-pad):cx+pad]
    
    # 2. Extract gradient magnitude (the sharp structural lines)
    gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    
    # 3. Threshold the strongest gradients (the actual physical edges of the 55mm box)
    _, edge_mask = cv2.threshold((mag / mag.max() * 255).astype(np.uint8), 40, 255, cv2.THRESH_BINARY)
    
    # 4. Bind the active gradients to measure total pixel width
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("[ERROR] Failed to detect target gradients for calibration.")
        
    all_pts = np.vstack(contours)
    _, _, w, h = cv2.boundingRect(all_pts)
    
    target_pixel_size = max(w, h)
    mm_per_px = checkerboard_size_mm / target_pixel_size
    
    return mm_per_px