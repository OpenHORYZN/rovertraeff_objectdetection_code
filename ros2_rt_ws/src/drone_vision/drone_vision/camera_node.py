"""Camera node - publishes frames from USB camera."""

import cv2
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from pathlib import Path
import sys
import os

# Add config to path - tries multiple locations
config_paths = [
    '/ros2_ws/config.py',
    os.path.expanduser('~/Documents/RoverTräff/rover-traeff/config.py'),
    Path(__file__).resolve().parents[3] / 'config.py',
]

config_dir = None
for p in config_paths:
    if Path(p).exists():
        config_dir = str(Path(p).parent)
        break

if config_dir:
    sys.path.insert(0, config_dir)

from config import config


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        
        # Get camera config
        camera_cfg = config["camera"]
        camera_model = camera_cfg["model"]  # "gopro", "d455", or "usb"
        device = camera_cfg["device"]
        width, height = camera_cfg["resolution"]
        fps = camera_cfg["fps"]
        
        # Open camera
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {device}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Load intrinsics from camera_intrinsic.yaml
        yaml_path = Path(__file__).parent.parent.parent.parent / 'positioning' / 'camera_intrinsic.yaml'
        with open(yaml_path) as f:
            cam_data = yaml.safe_load(f)
        
        # Get intrinsics for this camera model
        if camera_model == "gopro":
            cam_config = cam_data.get("gopro", {})
        else:
            cam_config = cam_data  # D455 is at root level
        
        # Extract intrinsics and distortion
        intr = cam_config.get("intrinsic_matrix", {})
        dist = cam_config.get("distortion", {})
        
        self.K = np.array([
            [intr.get("fx", width), 0, intr.get("cx", width/2)],
            [0, intr.get("fy", height), intr.get("cy", height/2)],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.D = np.array([
            dist.get("k1", 0), dist.get("k2", 0), dist.get("p1", 0),
            dist.get("p2", 0), dist.get("k3", 0)
        ], dtype=np.float32)
        
        # Publishers
        self.pub_image = self.create_publisher(Image, '/camera/color/image_raw', 1)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/color/camera_info', 1)
        self.bridge = CvBridge()
        
        # Camera info message
        self.camera_info = CameraInfo()
        self.camera_info.header.frame_id = 'camera_link'
        self.camera_info.height = height
        self.camera_info.width = width
        self.camera_info.k = self.K.flatten().tolist()
        self.camera_info.d = self.D.tolist()
        
        # Timer
        self.create_timer(1.0 / fps, self.timer_callback)
        self.get_logger().info(f"Camera: {camera_model} @ {width}x{height} ({fps}Hz)")
    
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Publish image
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.frame_id = 'camera_link'
        self.camera_info.header = msg.header
        
        self.pub_image.publish(msg)
        self.pub_info.publish(self.camera_info)

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        rclpy.shutdown()

if __name__ == '__main__':
    main()