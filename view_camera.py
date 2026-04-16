#!/usr/bin/env python3
"""Simple viewer: Display ROS camera stream on your computer."""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        self.sub = self.create_subscription(Image, '/camera/color/image_raw', self.callback, 1)
        self.bridge = CvBridge()
    
    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('GoPro Live Feed', frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    viewer = CameraViewer()
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
