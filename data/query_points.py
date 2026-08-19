"""
Query points setup: Structure Tensor Saddle Detection & Kinematic Rigidity Sorting.

This module replaces legacy OpenCV checkerboard detection and manual background 
sampling with physics-based extraction, guaranteeing zero human variance.
"""
import numpy as np
import cv2

def detect_all_saddle_points(frame_gray, expected_points=3):
    """
    Uses Structure Tensor eigenvalues (Shi-Tomasi) to find the strict 
    mathematical saddle points, ignoring edges, 1D lines, and background noise.
    
    Args:
        frame_gray: The grayscale reference frame (usually frame 0).
        expected_points: The number of physical 2x2 targets in the scene.
        
    Returns:
        corners: (N, 2) numpy array of sub-pixel coordinates.
    """
    # 1. Integer Guess using Eigenvalues
    corners = cv2.goodFeaturesToTrack(
        frame_gray,
        maxCorners=expected_points,
        qualityLevel=0.01, # Strict eigenvalue threshold to ignore floor weld lines
        minDistance=150,   # Ensure we don't pick two points on the same tripod
        useHarrisDetector=False
    )
    
    if corners is None or len(corners) < expected_points:
        raise ValueError(f"[ERROR] Failed to find {expected_points} valid 2x2 saddle points. Check image contrast.")
        
    # 2. Mathematical Sub-Pixel Refinement using Image Gradients
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(frame_gray, corners, (11, 11), (-1, -1), criteria)
    
    return corners.reshape(-1, 2)


def sort_by_kinematic_rigidity(frames, detected_saddle_points):
    """
    Universally identifies the vibrating target among reference points by 
    calculating pairwise distance variance over a short 50-frame pre-flight window.
    
    Args:
        frames: A list/array of the first 50 grayscale video frames.
        detected_saddle_points: The (3, 2) array of coordinates to test.
        
    Returns:
        target_point: The (2,) sub-pixel coordinate of the vibrating cable.
        bg_points: A list of (2,) coordinates for the rigidly mounted references.
    """
    if len(detected_saddle_points) != 3:
        raise ValueError(f"[ERROR] Kinematic sorter expects exactly 3 points, got {len(detected_saddle_points)}.")

    # 1. Take a 1-second slice of the video (50 frames)
    pre_flight_frames = frames[:50]
    p0 = np.array(detected_saddle_points, dtype=np.float32).reshape(-1, 1, 2)
    trajectories = [p0.reshape(3, 2)]
    
    # 2. Rapid Optical Flow tracking for pre-flight test
    old_gray = pre_flight_frames[0]
    for frame_gray in pre_flight_frames[1:]:
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, 
            winSize=(21, 21), maxLevel=2
        )
        trajectories.append(p1.reshape(3, 2))
        old_gray = frame_gray.copy()
        p0 = p1
        
    trajectories = np.array(trajectories)
    
    # 3. Calculate pairwise physical distances over time
    dist_0_1 = np.linalg.norm(trajectories[:, 0] - trajectories[:, 1], axis=1)
    dist_0_2 = np.linalg.norm(trajectories[:, 0] - trajectories[:, 2], axis=1)
    dist_1_2 = np.linalg.norm(trajectories[:, 1] - trajectories[:, 2], axis=1)
    
    # 4. Rank by relative variance (flatness of the distance line)
    variances = [
        (np.var(dist_0_1), 0, 1), 
        (np.var(dist_0_2), 0, 2), 
        (np.var(dist_1_2), 1, 2)
    ]
    variances.sort(key=lambda x: x[0])
    
    # 5. The pair with the absolute minimum variance are the rigid floor tripods
    ref_idx_1 = variances[0][1]
    ref_idx_2 = variances[0][2]
    
    # 6. The remaining point is the independent vibrating cable target
    target_idx = [i for i in [0, 1, 2] if i not in (ref_idx_1, ref_idx_2)][0]
    
    target_point = detected_saddle_points[target_idx]
    bg_points = [detected_saddle_points[ref_idx_1], detected_saddle_points[ref_idx_2]]
    
    return target_point, bg_points


# ==========================================
# Standalone Test Execution
# ==========================================
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.video import load_video_frames
    from paths import video_path

    # resolved via paths.py (env var or configs/paths.json) - see README
    video = video_path()
    print(f"[INFO] Loading pre-flight frames from {video}...")

    frames, _ = load_video_frames(video, max_frames=50, grayscale=True)
    
    print("[INFO] Detecting mathematical saddle points...")
    raw_saddles = detect_all_saddle_points(frames[0], expected_points=3)
    
    print("[INFO] Executing Kinematic Rigidity Test (50 frames)...")
    target_pt, bg_pts = sort_by_kinematic_rigidity(frames, raw_saddles)
    
    print("\n" + "="*50)
    print("EXTRACTION RESULTS")
    print("="*50)
    print(f"Target (Vibrating Cable): X={target_pt[0]:.2f}, Y={target_pt[1]:.2f}")
    for i, pt in enumerate(bg_pts):
        print(f"Background Ref {i} (Floor): X={pt[0]:.2f}, Y={pt[1]:.2f}")