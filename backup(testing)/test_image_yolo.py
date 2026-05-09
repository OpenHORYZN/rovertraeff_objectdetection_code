#!/usr/bin/env python3
"""
Offline test: Single image YOLO + ArUco detection + 3D pose estimation.

Tests a single saved image (from test set) and outputs:
- Annotated image with YOLO boxes + ArUco markers
- CSV with detections including 3D world coordinates

Usage:
    python3 test_image_yolo.py [image_path] [model_path]
"""

import csv
import json

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # one folder above
from config import config



def make_transform(rotation_matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Create 4x4 homogeneous transform from rotation matrix and translation."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform


def load_camera_intrinsics():
    """Load camera intrinsics from positioning/camera_intrinsic.yaml."""
    intrinsic_path = Path(__file__).parent / "positioning" / "camera_intrinsic.yaml"
    if not intrinsic_path.exists():
        raise FileNotFoundError(f"Camera intrinsics not found: {intrinsic_path}")
    
    with open(intrinsic_path, "r") as f:
        data = yaml.safe_load(f)
    
    intr = data.get("intrinsic_matrix", {})
    dist = data.get("distortion", {})
    
    K = np.array([
        [intr.get("fx"), 0, intr.get("cx")],
        [0, intr.get("fy"), intr.get("cy")],
        [0, 0, 1]
    ], dtype=np.float64)
    
    D = np.array([
        dist.get("k1", 0), dist.get("k2", 0), dist.get("p1", 0),
        dist.get("p2", 0), dist.get("k3", 0)
    ], dtype=np.float64)
    
    return K, D


def load_marker_world_positions():
    """Load ArUco marker world coordinates from positioning/aruco_pos.yaml."""
    aruco_path = Path(__file__).parent / "positioning" / "aruco_pos.yaml"
    if not aruco_path.exists():
        raise FileNotFoundError(f"ArUco positions not found: {aruco_path}")
    
    with open(aruco_path, "r") as f:
        data = yaml.safe_load(f)
    
    marker_positions = {}
    for marker in data.get("markers", []):
        marker_id = int(marker["id"])
        marker_positions[marker_id] = np.array(
            [float(marker["x"]), float(marker["y"]), float(marker.get("z", 0.0))],
            dtype=np.float64
        )
    
    return marker_positions


def detect_aruco_markers(frame, K, D, marker_length=0.15, dictionary_name="DICT_5X5_250"):
    """Detect ArUco markers and estimate their pose.

    Returns frame, marker_poses (rvec/tvec) and marker_corners (image corner coordinates per id).
    """
    dictionary_id = getattr(cv2.aruco, dictionary_name, cv2.aruco.DICT_5X5_250)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    
    # Draw markers
    if ids is not None and len(ids) > 0:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    marker_poses = {}
    marker_corners = {}
    if ids is not None and len(ids) > 0 and K is not None and D is not None:
        rvecs, tvecs = [], []
        for i, corner in enumerate(corners):
            # Define 3D marker corners (marker centered at origin in marker frame)
            obj_points = np.array([
                [-marker_length / 2, marker_length / 2, 0],
                [marker_length / 2, marker_length / 2, 0],
                [marker_length / 2, -marker_length / 2, 0],
                [-marker_length / 2, -marker_length / 2, 0]
            ], dtype=np.float32)
            
            # Solve PnP to get rotation and translation
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                corner[0],
                K,
                D,
                useExtrinsicGuess=False,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
        
        for i, marker_id in enumerate(ids.flatten()):
            # save image-space corners for each marker id
            try:
                marker_corners[int(marker_id)] = corners[i][0].astype(np.float32)
            except Exception:
                pass

            if i < len(rvecs):
                marker_poses[int(marker_id)] = {
                    "rvec": rvecs[i],
                    "tvec": tvecs[i],
                }
                
                # Draw axes for visualization
                try:
                    cv2.drawFrameAxes(frame, K, D, rvecs[i], tvecs[i], marker_length * 0.5)
                except cv2.error:
                    pass
    
    return frame, marker_poses, marker_corners


def estimate_world_position(detection, K, D, marker_poses, marker_world_positions, marker_corners, marker_length=0.15, reference_marker_id=None):
    """Estimate 3D world position of a YOLO detection using visible ArUco markers.

    Strategy:
    - If multiple markers are visible, build many 3D-2D correspondences (marker corners in world coords -> image corners)
      and solve for camera pose with `cv2.solvePnPRansac` to obtain camera->world transform.
    - If only a single marker is visible, fall back to the previous single-marker method.
    - Intersect the camera ray through detection center with the world plane (Z = marker Z) to get 2D world X,Y.
    """
    center = detection.get("center")
    if center is None or K is None:
        return None, None

    center_x, center_y = center

    # Build correspondences across all visible markers
    obj_pts_list = []
    img_pts_list = []
    for mid, corners_img in marker_corners.items():
        if mid not in marker_world_positions:
            continue
        mw = marker_world_positions[mid]
        # marker corners in world coordinates (centered)
        offsets = np.array([
            [-marker_length / 2, marker_length / 2, 0.0],
            [marker_length / 2, marker_length / 2, 0.0],
            [marker_length / 2, -marker_length / 2, 0.0],
            [-marker_length / 2, -marker_length / 2, 0.0]
        ], dtype=np.float64)
        obj = (mw.reshape(1, 3) + offsets).astype(np.float32)
        img = np.array(corners_img, dtype=np.float32)
        obj_pts_list.append(obj)
        img_pts_list.append(img)

    if len(obj_pts_list) >= 1:
        obj_pts = np.vstack(obj_pts_list)
        img_pts = np.vstack(img_pts_list)

        if obj_pts.shape[0] >= 4:
            # Solve for camera pose in world coordinates (world -> camera)
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                # fallback to single-marker method if available
                success = False
            else:
                # build transform_world_camera by inverting camera transform
                rot_cam_world, _ = cv2.Rodrigues(rvec)
                transform_camera_world = make_transform(rot_cam_world, tvec.flatten())
                transform_world_camera = np.linalg.inv(transform_camera_world)

                R_world_cam = transform_world_camera[:3, :3]
                T_world_cam = transform_world_camera[:3, 3]

                # plane Z: use average marker Z if available else 0
                plane_z = float(np.mean([marker_world_positions[mid][2] for mid in marker_corners.keys() if mid in marker_world_positions])) if len(marker_corners) > 0 else 0.0

                fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
                d_cam = np.array([ (center_x - cx) / fx, (center_y - cy) / fy, 1.0 ], dtype=np.float64)
                R_row3 = R_world_cam[2, :]
                denom = float(R_row3.dot(d_cam))
                if abs(denom) < 1e-6:
                    return None, T_world_cam
                s = (plane_z - float(T_world_cam[2])) / denom
                if s <= 0:
                    return None, T_world_cam
                cam_point = s * d_cam
                world_point = R_world_cam.dot(cam_point) + T_world_cam
                # drone_point = camera position in world coords (T_world_cam)
                return world_point, T_world_cam

    # Fallback: use reference marker if available
    if reference_marker_id is None:
        return None, None
    if reference_marker_id not in marker_poses or reference_marker_id not in marker_world_positions:
        return None, None

    marker_pose = marker_poses[reference_marker_id]
    marker_world = np.array(marker_world_positions[reference_marker_id], dtype=np.float64)
    rvec = marker_pose["rvec"]
    tvec = marker_pose["tvec"]
    marker_rotation, _ = cv2.Rodrigues(rvec)
    transform_camera_marker = make_transform(marker_rotation, tvec)
    transform_world_marker = np.eye(4, dtype=np.float64)
    transform_world_marker[:3, 3] = marker_world
    transform_world_camera = transform_world_marker @ np.linalg.inv(transform_camera_marker)

    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    d_cam = np.array([ (center_x - cx) / fx, (center_y - cy) / fy, 1.0 ], dtype=np.float64)
    R_world_cam = transform_world_camera[:3, :3]
    T_world_cam = transform_world_camera[:3, 3]
    plane_z = float(marker_world[2]) if marker_world.shape[0] >= 3 else 0.0
    R_row3 = R_world_cam[2, :]
    denom = float(R_row3.dot(d_cam))
    if abs(denom) < 1e-6:
        return None, T_world_cam
    s = (plane_z - float(T_world_cam[2])) / denom
    if s <= 0:
        return None, T_world_cam
    cam_point = s * d_cam
    world_point = R_world_cam.dot(cam_point) + T_world_cam
    return world_point, T_world_cam


def main():
    yolo_cfg = config["yolo"]
    camera_cfg = config["camera"]
    aruco_cfg = config.get("aruco", {})
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        image_path = Path("/home/conni/Documents/RoverTräff/rover-traeff/data/labeled/Resultslabeling/RbflwFT1TVT/test/images/rosbag2_2026_04_24-12_07_39_mp4-0018_jpg.rf.0cb63ef9a79c4069cb802d533611befe.jpg")
    
    if len(sys.argv) > 2:
        model_path = Path(sys.argv[2])
    else:
        model_path = Path(__file__).parent / "data" / "runs" / "FinalLabel_yolo11" / "weights" / "best.pt"
    
    print(f"\nImage: {image_path}")
    print(f"Model: {model_path}")
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return
    
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Failed to load image: {image_path}")
        return
    
    print(f"Image size: {frame.shape}")
    
    # Load models and intrinsics
    print("Loading YOLO model...")
    model = YOLO(str(model_path))
    model.to(yolo_cfg["device"])
    
    print("Loading camera intrinsics...")
    try:
        K, D = load_camera_intrinsics()
        print(f"Camera matrix K:\n{K}")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        K, D = None, None
    
    print("Loading marker world positions...")
    try:
        marker_world_positions = load_marker_world_positions()
        print(f"Found {len(marker_world_positions)} markers")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        marker_world_positions = {}
    
    marker_length = float(aruco_cfg.get("marker_length", 0.15))
    dictionary_name = aruco_cfg.get("dictionary", "DICT_5X5_250")
    reference_marker_id = int(aruco_cfg.get("reference_marker_id", 101))
    
    # Run YOLO detection
    print("\nRunning YOLO detection...")
    results = model.predict(
        frame,
        conf=yolo_cfg["confidence"],
        imgsz=yolo_cfg["imgsz"],
        verbose=False,
    )
    
    print(f"Detected {len(results[0].boxes)} objects")
    annotated = results[0].plot()
    
    # Detect ArUco markers
    print("Detecting ArUco markers...")
    annotated, marker_poses, marker_corners = detect_aruco_markers(annotated, K, D, marker_length, dictionary_name)
    print(f"Found {len(marker_poses)} markers: {list(marker_poses.keys())}")
    
    # Process detections
    print("\nDetections:")
    detections_list = []
    
    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        class_name = str(model.names.get(class_id, class_id))
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        confidence = float(box.conf[0].item())
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        
        detection = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2],
            "center": [center_x, center_y],
        }
        
        world_point, drone_point = estimate_world_position(
            detection, K, D, marker_poses, marker_world_positions, marker_corners, marker_length, reference_marker_id
        )
        
        print(f"  {class_name} ({confidence:.2f}) at ({center_x:.1f}, {center_y:.1f})")
        if world_point is not None:
            print(f"    World: ({world_point[0]:.3f}, {world_point[1]:.3f})")
            print(f"    Drone: ({drone_point[0]:.3f}, {drone_point[1]:.3f}, {drone_point[2]:.3f})")
        else:
            print(f"    Position: Could not estimate (missing markers or intrinsics)")
        
        detections_list.append({
            "class_name": class_name,
            "confidence": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": center_x,
            "center_y": center_y,
            "world_x": world_point[0] if world_point is not None else None,
            "world_y": world_point[1] if world_point is not None else None,
            "drone_x": drone_point[0] if drone_point is not None else None,
            "drone_y": drone_point[1] if drone_point is not None else None,
        })
    
    # Save annotated image
    output_dir = Path(__file__).parent / "data" / "detections" / "test_image_yolo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_image_path = output_dir / f"{image_path.stem}_annotated.jpg"
    cv2.imwrite(str(output_image_path), annotated)
    print(f"\nAnnotated image saved: {output_image_path}")
    
    # Save CSV
    csv_path = output_dir / "detections.csv"
    csv_columns = [
        "class_name", "confidence", "x1", "y1", "x2", "y2",
        "center_x", "center_y",
        "world_x", "world_y",
        "drone_x", "drone_y"
    ]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for det in detections_list:
            writer.writerow(det)
    
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()