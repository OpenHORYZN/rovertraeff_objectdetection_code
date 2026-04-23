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

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from config import config


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        
        # Get camera config
        camera_cfg = config["camera"]
        camera_model = camera_cfg["model"]  # "phone", "d455"
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
        
        # Load intrinsics from camera_intrinsic.yaml, or use a simple fallback for phone streams.
        yaml_path = Path(__file__).parent.parent.parent.parent / 'positioning' / 'camera_intrinsic.yaml'
        if camera_model == "phone":
            intr = {"fx": float(width), "fy": float(width), "cx": width / 2.0, "cy": height / 2.0}
            dist = {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0}
            self.get_logger().warn("Using approximate intrinsics for phone camera")
        else:
            with open(yaml_path) as f:
                cam_data = yaml.safe_load(f)
                cam_config = cam_data  

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
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()