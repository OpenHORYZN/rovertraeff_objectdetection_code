#!/usr/bin/env python3
"""Aruco node - detects markers and publishes marker id + rvec/tvec + annotated image."""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Int32

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


class ArucoNode(Node):
    def __init__(self):
        super().__init__('aruco_node')

        camera_cfg = config['camera']
        aruco_cfg = config.get('aruco', {})
        self.marker_length = float(aruco_cfg.get('marker_length', 0.15))
        self.dictionary_name = aruco_cfg.get('dictionary', 'DICT_5X5_250')
        dictionary_id = getattr(cv2.aruco, self.dictionary_name, cv2.aruco.DICT_5X5_250)
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

        self.camera_matrix = None
        self.dist_coeffs = None

        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 1)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_callback, 1)

        self.annotated_pub = self.create_publisher(Image, '/aruco/annotated', 1)
        self.rvec_pub = self.create_publisher(Vector3, '/aruco/rvec', 1)
        self.tvec_pub = self.create_publisher(Vector3, '/aruco/tvec', 1)
        self.marker_id_pub = self.create_publisher(Int32, '/aruco/marker_id', 1)

        self.bridge = CvBridge()
        self.get_logger().info(
            f'Aruco ready. dictionary={self.dictionary_name}, marker_length={self.marker_length} m, camera={camera_cfg["model"]}'
        )

    def info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if len(msg.d) >= 5:
            self.dist_coeffs = np.array(msg.d[:5], dtype=np.float64)
        else:
            self.dist_coeffs = np.zeros(5, dtype=np.float64)

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            self.annotated_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))
            return

        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        if self.camera_matrix is not None and self.dist_coeffs is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.marker_length,
                self.camera_matrix,
                self.dist_coeffs,
            )

            first_id = int(ids[0][0])
            first_rvec = rvecs[0][0]
            first_tvec = tvecs[0][0]

            try:
                cv2.drawFrameAxes(
                    frame,
                    self.camera_matrix,
                    self.dist_coeffs,
                    first_rvec,
                    first_tvec,
                    self.marker_length * 0.5,
                )
            except cv2.error:
                pass

            rvec_msg = Vector3()
            rvec_msg.x = float(first_rvec[0])
            rvec_msg.y = float(first_rvec[1])
            rvec_msg.z = float(first_rvec[2])
            self.rvec_pub.publish(rvec_msg)

            tvec_msg = Vector3()
            tvec_msg.x = float(first_tvec[0])
            tvec_msg.y = float(first_tvec[1])
            tvec_msg.z = float(first_tvec[2])
            self.tvec_pub.publish(tvec_msg)

            marker_id_msg = Int32()
            marker_id_msg.data = first_id
            self.marker_id_pub.publish(marker_id_msg)

            self.get_logger().info(
                f'Detected marker {first_id} tvec=({first_tvec[0]:.2f}, {first_tvec[1]:.2f}, {first_tvec[2]:.2f})'
            )

        self.annotated_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()