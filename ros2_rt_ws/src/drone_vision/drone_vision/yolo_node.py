#!/usr/bin/env python3
"""YOLO detection node - subscribes to camera, detects objects, publishes detections."""

import json
from pathlib import Path
import sys
import time

import cv2
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

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

class YOLONode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        
        # Load YOLO model
        yolo_cfg = config["yolo"]
        model_path = yolo_cfg.get("modellive")
        base_dir = Path(config["paths"]["base"])
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = base_dir / model_path
        confidence = yolo_cfg["confidence"]
        device = yolo_cfg["device"]
        imgsz = int(yolo_cfg.get("imgsz"))
        desired_classes = yolo_cfg.get("classes", [])
        
        self.get_logger().info(f"Loading YOLO: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        self.confidence = confidence
        self.device = device
        self.imgsz = imgsz
        self.track_distance_px = float(yolo_cfg.get('track_distance_px', 80.0))
        self.track_timeout_s = float(yolo_cfg.get('track_timeout_s', 1.5))
        self.next_track_id = 1
        self.active_tracks = []

        self.class_ids = None
        if desired_classes:
            name_to_id = {str(v): int(k) for k, v in self.model.names.items()}
            mapped_ids = [name_to_id[name] for name in desired_classes if name in name_to_id]
            if mapped_ids:
                self.class_ids = mapped_ids
            self.get_logger().info(f"Model classes: {name_to_id}")
            self.get_logger().info(f"Desired classes: {desired_classes}")
            self.get_logger().info(f"Class IDs: {mapped_ids}")
        
        # Subscribe to camera frames
        self.sub = self.create_subscription(Image, '/main_mission/aligned_image', self.detect, 1)
        self.bridge = CvBridge()
        
        # Publisher for annotated images (optional)
        self.pub = self.create_publisher(Image, '/detections/annotated', 1)
        self.raw_pub = self.create_publisher(String, '/detections/raw', 1)

        # Simple photo logging for report generation.
        self.photo_dir = base_dir / 'data' / 'detections' / 'ros_yolo' / 'images'
        self.photo_dir.mkdir(parents=True, exist_ok=True)
        
        self.get_logger().info(
            f"YOLO ready. Confidence: {confidence}, Device: {device}, imgsz: {imgsz}, classes: {self.class_ids}"
        )
    
    def detect(self, msg):
        """Run YOLO on incoming frame."""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        now = time.time()
        
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            classes=self.class_ids,
            verbose=False,
        )
        
        # Annotate frame
        annotated = results[0].plot()
        
        # Publish annotated frame
        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)

        # Publish raw detections for downstream fusion nodes.
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            class_name = str(self.model.names.get(class_id, class_id))
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            confidence = float(box.conf[0].item())
            center_x = 0.5 * (x1 + x2)
            center_y = 0.5 * (y1 + y2)

            # Save one full-frame image per tracked probe instead of one crop per frame.
            self._prune_stale_tracks(now)
            track = self._match_track(class_name, center_x, center_y, now)
            image_path = ''
            if track is not None and track['image_path']:
                image_path = track['image_path']
            else:
                ts = time.strftime('%Y-%m-%dT%H-%M-%S', time.localtime(now))
                suffix = int((now - int(now)) * 1000)
                track_id = track['id'] if track is not None else self.next_track_id
                image_name = f'{ts}_{suffix:03d}_{class_name}_track{track_id:03d}_{confidence:.2f}.jpg'
                full_path = self.photo_dir / image_name
                cv2.imwrite(str(full_path), frame)
                image_path = str(full_path)

                if track is None:
                    track = {
                        'id': int(self.next_track_id),
                        'class_name': class_name,
                        'center_x': float(center_x),
                        'center_y': float(center_y),
                        'last_seen': float(now),
                        'image_path': image_path,
                    }
                    self.active_tracks.append(track)
                    self.next_track_id += 1
                else:
                    track['image_path'] = image_path

            if track is not None:
                track['center_x'] = float(center_x)
                track['center_y'] = float(center_y)
                track['last_seen'] = float(now)

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "center": [0.5 * (x1 + x2), 0.5 * (y1 + y2)],
                    "image_path": image_path,
                }
            )

        raw_msg = String()
        raw_msg.data = json.dumps(
            {
                "frame_id": msg.header.frame_id,
                "stamp": {
                    "sec": int(msg.header.stamp.sec),
                    "nanosec": int(msg.header.stamp.nanosec),
                },
                "detections": detections,
            }
        )
        self.raw_pub.publish(raw_msg)
        
        # Log detections
        if len(results[0].boxes) > 0:
            self.get_logger().info(f"Detected {len(results[0].boxes)} objects")

    def _prune_stale_tracks(self, now: float):
        self.active_tracks = [
            track for track in self.active_tracks if now - track['last_seen'] <= self.track_timeout_s
        ]

    def _match_track(self, class_name: str, center_x: float, center_y: float, now: float):
        best_track = None
        best_distance_sq = self.track_distance_px * self.track_distance_px
        for track in self.active_tracks:
            if track['class_name'] != class_name:
                continue
            if now - track['last_seen'] > self.track_timeout_s:
                continue
            dx = track['center_x'] - center_x
            dy = track['center_y'] - center_y
            distance_sq = dx * dx + dy * dy
            if distance_sq <= best_distance_sq:
                best_distance_sq = distance_sq
                best_track = track
        return best_track


def main(args=None):
    rclpy.init(args=args)
    node = YOLONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
