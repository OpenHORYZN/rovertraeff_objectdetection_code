#!/usr/bin/env python3
"""Fuse ArUco pose and YOLO detections into approximate world positions."""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Int32
import yaml
from std_msgs.msg import String

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
        self.reference_marker_id = int(config.get('aruco', {}).get('reference_marker_id', 101))
        self.base_dir = Path(config['paths']['base'])

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
        self.latest_rvec = None
        self.latest_tvec = None
        self.latest_marker_id = None

        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 1)
        self.create_subscription(Vector3, '/aruco/rvec', self.rvec_callback, 1)
        self.create_subscription(Vector3, '/aruco/tvec', self.tvec_callback, 1)
        self.create_subscription(Int32, '/aruco/marker_id', self.marker_id_callback, 1)
        self.create_subscription(String, '/detections/raw', self.detections_callback, 1)

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
            f'Pose estimator ready. target_classes={sorted(self.target_classes) if self.target_classes else "all"}, reference_marker_id={self.reference_marker_id}'
        )

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if len(msg.d) >= 5:
            self.dist_coeffs = np.array(msg.d[:5], dtype=np.float64)
        else:
            self.dist_coeffs = np.zeros(5, dtype=np.float64)

    def rvec_callback(self, msg: Vector3):
        self.latest_rvec = np.array([msg.x, msg.y, msg.z], dtype=np.float64)

    def tvec_callback(self, msg: Vector3):
        self.latest_tvec = np.array([msg.x, msg.y, msg.z], dtype=np.float64)

    def marker_id_callback(self, msg: Int32):
        self.latest_marker_id = int(msg.data)

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
                continue

            estimated_points = self._estimate_points(detection)
            if estimated_points is None:
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

    def _estimate_points(self, detection):
        # 2) Require the minimum data needed for geometry.
        if self.camera_matrix is None or self.dist_coeffs is None:
            return None
        if self.latest_rvec is None or self.latest_tvec is None or self.latest_marker_id is None:
            return None

        marker_world = self.marker_world_positions.get(self.latest_marker_id)
        if marker_world is None:
            return None

        # 3) Build camera->marker and world->camera transforms from rvec/tvec + known marker world position.
        marker_rotation, _ = cv2.Rodrigues(self.latest_rvec)
        marker_translation = self.latest_tvec

        transform_camera_marker = make_transform(marker_rotation, marker_translation)
        transform_world_marker = np.eye(4, dtype=np.float64)
        transform_world_marker[:3, 3] = marker_world
        transform_world_camera = transform_world_marker @ np.linalg.inv(transform_camera_marker)

        center = detection.get('center', None)
        if center is None:
            bbox = detection.get('bbox', None)
            if bbox is None:
                return None
            u = 0.5 * (float(bbox[0]) + float(bbox[2]))
            v = 0.5 * (float(bbox[1]) + float(bbox[3]))
        else:
            u, v = float(center[0]), float(center[1])

        # 4) Undistort pixel and convert to a normalized camera ray.
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        pixel = np.array([[[u, v]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(pixel, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        u_corr = float(undistorted[0, 0, 0])
        v_corr = float(undistorted[0, 0, 1])

        ray_camera = np.array([(u_corr - cx) / fx, (v_corr - cy) / fy, 1.0], dtype=np.float64)
        ray_camera /= np.linalg.norm(ray_camera)

        # 5) Intersect camera ray with the marker plane in camera frame.
        plane_point_camera = marker_translation
        plane_normal_camera = marker_rotation[:, 2]
        denom = float(np.dot(plane_normal_camera, ray_camera))
        if abs(denom) < 1e-8:
            return None

        scale_camera = float(np.dot(plane_normal_camera, plane_point_camera) / denom)
        if scale_camera <= 0.0:
            return None

        drone_point = scale_camera * ray_camera

        # 6) Transform the camera-frame point to world frame.
        rotation_world_camera = transform_world_camera[:3, :3]
        origin_world = transform_world_camera[:3, 3]
        world_point = origin_world + (rotation_world_camera @ drone_point)
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