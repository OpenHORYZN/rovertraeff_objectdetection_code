#!/usr/bin/env python3
"""Fuse ArUco pose and YOLO detections into approximate world positions."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Vector3
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Int32, String
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from config import config


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

        self.world_pub = self.create_publisher(PointStamped, '/detections_world', 1)
        self.world_raw_pub = self.create_publisher(String, '/detections_world/raw', 1)

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

        detections_world = []
        for detection in detections:
            class_name = detection.get('class_name', '')
            if self.target_classes and class_name not in self.target_classes:
                continue

            world_point = self._estimate_world_point(detection)
            if world_point is None:
                continue

            # Publish the point for simple ROS visualization.
            out_msg = PointStamped()
            out_msg.header.stamp = self.get_clock().now().to_msg()
            out_msg.header.frame_id = 'world'
            out_msg.point.x = float(world_point[0])
            out_msg.point.y = float(world_point[1])
            out_msg.point.z = float(world_point[2])
            self.world_pub.publish(out_msg)

            detections_world.append(
                {
                    'class_name': class_name,
                    'confidence': float(detection.get('confidence', 0.0)),
                    'image_path': detection.get('image_path', ''),
                    'world': [float(world_point[0]), float(world_point[1]), float(world_point[2])],
                }
            )

        if detections_world:
            raw_msg = String()
            raw_msg.data = json.dumps({'detections_world': detections_world})
            self.world_raw_pub.publish(raw_msg)

    def _estimate_world_point(self, detection):
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

        # 4) Pick the 2D pixel center from detection bbox.
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        center = detection.get('center', None)
        if center is None:
            bbox = detection.get('bbox', None)
            if bbox is None:
                return None
            u = 0.5 * (float(bbox[0]) + float(bbox[2]))
            v = 0.5 * (float(bbox[1]) + float(bbox[3]))
        else:
            u, v = float(center[0]), float(center[1])

        # 5) Undistort pixel and convert to a normalized camera ray.
        pixel = np.array([[[u, v]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(pixel, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        u_corr = float(undistorted[0, 0, 0])
        v_corr = float(undistorted[0, 0, 1])

        ray_camera = np.array([(u_corr - cx) / fx, (v_corr - cy) / fy, 1.0], dtype=np.float64)
        ray_camera /= np.linalg.norm(ray_camera)

        # 6) Transform ray from camera frame to world frame.
        rotation_world_camera = transform_world_camera[:3, :3]
        origin_world = transform_world_camera[:3, 3]
        ray_world = rotation_world_camera @ ray_camera

        # 7) Intersect world ray with ground plane z = marker_world_z.
        plane_z = float(marker_world[2])
        if abs(ray_world[2]) < 1e-8:
            return None

        scale = (plane_z - origin_world[2]) / ray_world[2]
        world_point = origin_world + scale * ray_world
        return world_point


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