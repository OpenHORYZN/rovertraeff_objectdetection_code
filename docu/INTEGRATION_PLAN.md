# Plan: Integrate Camera + YOLO + ArUco into ROS 2 Pipeline

## TL;DR
Build 4 ROS 2 nodes: camera source → YOLO detection + ArUco calibration → coordinate transformation → logged results with photos.

## Key Simplifications:
- **No depth needed**: Objects on floor (z=0), only need (x,y) world coords
- **2D projection only**: Pixel (u,v) → camera ray → intersect z=0 plane → world (x,y)
- **Reuse px4 container**: Add vision nodes alongside existing PX4/MAVROS
- **Generic camera support**: Works with GoPro now (USB), swap to D455 later

## Steps

### Phase 1: Setup (Foundation)
1. Add GoPro intrinsics to `positioning/camera_intrinsic.yaml` ✓ (done above)
2. Set up ROS 2 workspace structure (see Docker section below)
3. Verify ArUco positions in `positioning/aruco_pos.yaml`

### Phase 2: Build 4 ROS Nodes (can do in parallel)
2a. **camera_node** (generic USB camera)
  - Publishes: `/camera/color/image_raw`, `/camera/color/camera_info`
  - Uses pyv4l2 or cv2.VideoCapture (GoPro via USB)
  - ~15-30 Hz

2b. **yolo_detector_node**
  - Subscribes: `/camera/color/image_raw`
  - Publishes: `/detections` topic + saves photos to `data/detections/photos/`
  - Uses: `yolo11n.pt` (your trained model)

2c. **aruco_detector_node**
  - Subscribes: `/camera/color/image_raw`
  - Publishes: `/camera_pose` (camera position in world frame)
  - Uses: ArUco marker IDs + known positions from `positioning/aruco_pos.yaml`
  - IMPORTANT: Use cv2.undistort() first (GoPro barrel distortion)

### Phase 3: Coordinate Transform (depends on Phase 2)
3. **pose_estimator_node** (2D only)
  - Subscribes: `/detections`, `/camera_pose`, `/camera/color/camera_info`
  - For each detection bbox (u,v) in pixels:
    - Compute camera ray: ray = K^(-1)[u, v, 1]^T
    - Intersect with z=0: where does camera ray hit floor?
    - Transform to world: use camera pose R,t
    - Result: (x_world, y_world, z=0)
  - Publishes: `/detections_world` + CSV log

### Phase 4: Testing & Integration (depends on Phase 1-3)
4. Launch file: `ros2_ws/src/rover_vision/launch/rover_vision.launch.py`
5. Test each node individually
6. End-to-end verification

## Files to Create
- `ros2_ws/src/rover_vision/package.xml`
- `ros2_ws/src/rover_vision/rover_vision/camera_node.py`
- `ros2_ws/src/rover_vision/rover_vision/yolo_node.py`
- `ros2_ws/src/rover_vision/rover_vision/aruco_node.py`
- `ros2_ws/src/rover_vision/rover_vision/pose_estimator_node.py`
- `ros2_ws/src/rover_vision/launch/rover_vision.launch.py`

## Testing with GoPro
1. Connect GoPro via USB
2. Verify USB device: `ls -la /dev/video*`
3. Test camera_node: `ros2 run rover_vision camera_node`
4. Check topic: `ros2 topic echo /camera/color/image_raw` (should show image data)
5. Print ArUco markers, hold in view, verify `/camera_pose` updates
6. Place probe or rako box, verify `/detections_world` has valid (x,y) coordinates

## When D455 Arrives
- Replace camera_node to use pyrealsense2 instead of cv2.VideoCapture
- Recalibrate intrinsics with checkerboard
- All other nodes remain unchanged
