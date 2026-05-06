#!/usr/bin/env python3
"""Fuse ArUco pose and YOLO detections into approximate world positions."""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
import yaml
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseArray

# To use config.py
import os
import importlib.util
from ament_index_python.packages import get_package_share_directory

# Get the path to the share directory
package_share_dir = get_package_share_directory('drone_vision')
config_path = os.path.join(package_share_dir, 'config.py')

# Import the module dynamically
spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

# Now use it!
config = config_module.config

def make_transform(rotation_matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform


class PoseEstimatorNode(Node):
    def __init__(self):
        super().__init__('pose_estimator_node')

        self.yolo_cfg = config['yolo']
        configured_classes = self.yolo_cfg.get('classes') or []
        self.target_classes = set(configured_classes)
        self.base_dir = Path(config['paths']['base'])
        self.marker_length = float(config.get('aruco', {}).get('marker_length', 0.15))

        aruco_path = self.base_dir / 'positioning' / 'aruco_pos.yaml'
        with open(aruco_path, 'r', encoding='utf-8') as handle:
            aruco_data = yaml.safe_load(handle)

        self.marker_world_positions = {}
        for marker in aruco_data.get('markers', []):
            marker_id = int(marker['id'])
            self.marker_world_positions[marker_id] = np.array(
                [float(marker['x']), float(marker['y']), float(marker.get('z', 0.0))],
                dtype=np.float64,
            )

        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_markers = {}  # id -> {'rvec': [..], 'tvec': [..], 'corners': [[x,y],...]}

        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 1)
        self.create_subscription(String, '/aruco/markers', self.aruco_markers_callback, 1)
        self.create_subscription(String, '/detections/raw', self.detections_callback, 1)

        self.box_pub = self.create_publisher(PoseArray, '/boxes', 1)
        self.probe_pub = self.create_publisher(PoseArray, '/probes', 1)
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.boxes = PoseArray()
        self.probes = PoseArray()
        self.boxes.header.frame_id = 'map'
        self.probes.header.frame_id = 'map'

        self.log_points = bool(config.get('pose_estimator', {}).get('log_points', False))

        self.csv_path = self.base_dir / 'data' / 'detections' / 'ros_yolo' / 'detections.csv'
        self.csv_columns = [
            'timestamp',
            'class_name',
            'confidence',
            'x1',
            'y1',
            'x2',
            'y2',
            'center_x',
            'center_y',
            'drone_x',
            'drone_y',
            'world_x',
            'world_y',
            'image_path',
        ]
        self._ensure_csv_header()

        self.get_logger().info(
            f'Pose estimator ready. target_classes={sorted(self.target_classes) if self.target_classes else "all"}, multi-marker world estimation enabled'
        )

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if len(msg.d) >= 5:
            self.dist_coeffs = np.array(msg.d[:5], dtype=np.float64)
        else:
            self.dist_coeffs = np.zeros(5, dtype=np.float64)

    def aruco_markers_callback(self, msg: String):
        """Receive JSON list of visible markers published by aruco_node.

        Expected format: [{"id": int, "rvec": [x,y,z], "tvec": [x,y,z], "corners": [[x,y],...]}, ...]
        """
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        new = {}
        for m in data:
            try:
                mid = int(m.get('id'))
                new[mid] = {
                    'rvec': np.array(m.get('rvec', []), dtype=np.float64),
                    'tvec': np.array(m.get('tvec', []), dtype=np.float64),
                    'corners': np.array(m.get('corners', []), dtype=np.float32),
                }
            except Exception:
                continue
        self.latest_markers = new

    def detections_callback(self, msg: String):
        # 1) Parse incoming YOLO detections.
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('Failed to parse detection payload')
            return

        detections = payload.get('detections', [])
        if not detections:
            return

        stamp = payload.get('stamp', {})
        ts_sec = int(stamp.get('sec', 0))
        ts_nanosec = int(stamp.get('nanosec', 0))
        timestamp = ts_sec + (ts_nanosec * 1e-9)

        for detection in detections:
            class_name = detection.get('class_name', '')
            if self.target_classes and class_name not in self.target_classes:
                self.get_logger().warn('Detected class not in target classes')
                continue

            estimated_points = self._estimate_points(detection)
            if estimated_points is None:
                self.get_logger().warn('Point estimation failed')
                continue
            drone_point, world_point = estimated_points

            if self.log_points:
                self.get_logger().info(
                    f'{class_name}: drone=({drone_point[0]:.3f}, {drone_point[1]:.3f}) world=({world_point[0]:.3f}, {world_point[1]:.3f})'
                )

            bbox = detection.get('bbox', [None, None, None, None])
            if len(bbox) < 4:
                bbox = [None, None, None, None]
            center = detection.get('center', [None, None])
            if len(center) < 2:
                center = [None, None]

            self._append_csv_row(
                {
                    'timestamp': f'{timestamp:.6f}',
                    'class_name': class_name,
                    'confidence': f"{float(detection.get('confidence', 0.0)):.4f}",
                    'x1': self._fmt_float(bbox[0]),
                    'y1': self._fmt_float(bbox[1]),
                    'x2': self._fmt_float(bbox[2]),
                    'y2': self._fmt_float(bbox[3]),
                    'center_x': self._fmt_float(center[0]),
                    'center_y': self._fmt_float(center[1]),
                    'drone_x': f'{float(drone_point[0]):.4f}',
                    'drone_y': f'{float(drone_point[1]):.4f}',
                    'world_x': f'{float(world_point[0]):.4f}',
                    'world_y': f'{float(world_point[1]):.4f}',
                    'image_path': detection.get('image_path', ''),
                }
            )

            pose = Pose()
            pose.position.x = center[0]
            pose.position.y = center[1]
            match class_name:
                case 'rako':
                    self.boxes.poses.append(pose)
                    self.get_logger().info(f'Publishing box')
                case 'probes':
                    self.probes.poses.append(pose)
                    self.get_logger().info(f'Publishing probe')

    def timer_callback(self):
        self.box_pub.publish(self.boxes)
        self.probe_pub.publish(self.probes)

    def _estimate_points(self, detection):
        # 2) Require camera intrinsics.
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.get_logger().info(f'No intrinsics found')
            return None

        if not self.latest_markers:
            self.get_logger().info(f'No arucos detected yet')
            return None

        # Build 3D-2D correspondences from all visible markers
        obj_pts_list = []
        img_pts_list = []
        for mid, data in self.latest_markers.items():
            corners = data.get('corners')
            if corners is None or len(corners) < 4:
                self.get_logger().info(f'Malformed aruco corners')
                continue
            mw = self.marker_world_positions.get(int(mid)) # official position from yaml
            if mw is None:
                self.get_logger().info(f'Failed to get aruco sensor positions')
                continue
            offsets = np.array([
                [-0.5 * self.marker_length,  0.5 * self.marker_length, 0.0],
                [ 0.5 * self.marker_length,  0.5 * self.marker_length, 0.0],
                [ 0.5 * self.marker_length, -0.5 * self.marker_length, 0.0],
                [-0.5 * self.marker_length, -0.5 * self.marker_length, 0.0],
            ], dtype=np.float64)
            obj = (mw.reshape(1, 3) + offsets).astype(np.float32)
            obj_pts_list.append(obj)
            img_pts_list.append(np.array(corners, dtype=np.float32))

        if len(obj_pts_list) == 0:
            self.get_logger().info(f'No arucos in current detection')
            return None
        obj_pts = np.vstack(obj_pts_list)
        img_pts = np.vstack(img_pts_list)
        if obj_pts.shape[0] < 4:
            return None

        try:
            success, rvec, tvec, _ = cv2.solvePnPRansac(obj_pts, img_pts, self.camera_matrix, self.dist_coeffs)
        except Exception:
            self.get_logger().info(f'Exception while calculating aruco positions')
            return None
        if not success:
            self.get_logger().info(f'Exception while calculating aruco positions')
            return None

        rot_cam_world, _ = cv2.Rodrigues(rvec)
        transform_camera_world = make_transform(rot_cam_world, tvec.flatten())
        transform_world_camera = np.linalg.inv(transform_camera_world)

        # detection center
        center = detection.get('center', None)
        if center is None:
            bbox = detection.get('bbox', None)
            if bbox is None:
                return None
            u = 0.5 * (float(bbox[0]) + float(bbox[2]))
            v = 0.5 * (float(bbox[1]) + float(bbox[3]))
        else:
            u, v = float(center[0]), float(center[1])

        fx = self.camera_matrix[0, 0]; fy = self.camera_matrix[1, 1]; cx = self.camera_matrix[0, 2]; cy = self.camera_matrix[1, 2]
        d_cam = np.array([ (u - cx) / fx, (v - cy) / fy, 1.0 ], dtype=np.float64)

        R_world_cam = transform_world_camera[:3, :3]
        T_world_cam = transform_world_camera[:3, 3]
        plane_z = float(np.mean([self.marker_world_positions[int(mid)][2] for mid in self.latest_markers.keys() if int(mid) in self.marker_world_positions]))
        denom = float(R_world_cam[2, :].dot(d_cam))
        if abs(denom) < 1e-8:
            self.get_logger().info(f'Numerical error (denominator too small)')
            return None
        s = (plane_z - float(T_world_cam[2])) / denom
        if s <= 0.0:
            self.get_logger().info(f'Numerical error (s <= 0.0)')
            return None
        cam_point = s * d_cam
        world_point = R_world_cam.dot(cam_point) + T_world_cam
        drone_point = T_world_cam
        return drone_point, world_point

    def _fmt_float(self, value):
        if value is None:
            return ''
        return f'{float(value):.2f}'

    def _ensure_csv_header(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=self.csv_columns)
                writer.writeheader()
            return

        with self.csv_path.open('r', newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            existing_columns = reader.fieldnames or []
            rows = list(reader)

        if existing_columns == self.csv_columns:
            return

        with self.csv_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.csv_columns)
            writer.writeheader()
            for row in rows:
                migrated = {column: row.get(column, '') for column in self.csv_columns}
                writer.writerow(migrated)

    def _append_csv_row(self, row):
        with self.csv_path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.csv_columns)
            writer.writerow(row)


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()