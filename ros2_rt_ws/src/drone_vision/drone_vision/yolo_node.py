#!/usr/bin/env python3
"""YOLO detection node - subscribes to camera, detects objects, publishes detections."""

import csv
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

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from config import config


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

        self.class_ids = None
        if desired_classes:
            name_to_id = {str(v): int(k) for k, v in self.model.names.items()}
            mapped_ids = [name_to_id[name] for name in desired_classes if name in name_to_id]
            if mapped_ids:
                self.class_ids = mapped_ids
        
        # Subscribe to camera frames
        self.sub = self.create_subscription(Image, '/camera/color/image_raw', self.detect, 1)
        self.bridge = CvBridge()
        
        # Publisher for annotated images (optional)
        self.pub = self.create_publisher(Image, '/detections/annotated', 1)
        self.raw_pub = self.create_publisher(String, '/detections/raw', 1)

        # Simple photo logging for report generation.
        self.photo_dir = base_dir / 'data' / 'detections' / 'ros_yolo' / 'images'
        self.photo_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = base_dir / 'data' / 'detections' / 'ros_yolo' / 'detections.csv'
        if not self.csv_path.exists():
            with self.csv_path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    'timestamp',
                    'class_name',
                    'confidence',
                    'x1',
                    'y1',
                    'x2',
                    'y2',
                    'center_x',
                    'center_y',
                    'image_path',
                ])
        
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

            # Save one cropped image per detection event.
            pad_x = int(0.08 * (x2 - x1))
            pad_y = int(0.08 * (y2 - y1))
            h, w = frame.shape[:2]
            left = max(0, int(x1) - pad_x)
            top = max(0, int(y1) - pad_y)
            right = min(w, int(x2) + pad_x)
            bottom = min(h, int(y2) + pad_y)
            crop = frame[top:bottom, left:right]

            image_path = ''
            if crop.size > 0:
                ts = time.strftime('%Y-%m-%dT%H-%M-%S', time.localtime(now))
                suffix = int((now - int(now)) * 1000)
                image_name = f'{ts}_{suffix:03d}_{class_name}_{confidence:.2f}.jpg'
                full_path = self.photo_dir / image_name
                cv2.imwrite(str(full_path), crop)
                image_path = str(full_path)

                center_x = 0.5 * (x1 + x2)
                center_y = 0.5 * (y1 + y2)
                with self.csv_path.open('a', newline='', encoding='utf-8') as handle:
                    writer = csv.writer(handle)
                    writer.writerow([
                        f'{now:.6f}',
                        class_name,
                        f'{confidence:.4f}',
                        f'{x1:.2f}',
                        f'{y1:.2f}',
                        f'{x2:.2f}',
                        f'{y2:.2f}',
                        f'{center_x:.2f}',
                        f'{center_y:.2f}',
                        image_path,
                    ])

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
