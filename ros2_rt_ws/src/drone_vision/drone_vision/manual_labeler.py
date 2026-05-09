#!/usr/bin/env python3
"""Manual labeler — replaces YOLO with mouse clicks, feeds pose_estimator_node."""

import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import message_filters


class ManualLabeler(Node):
    def __init__(self):
        super().__init__('manual_labeler')

        self.bridge_  = CvBridge()
        self.clicks_  = []   # list of (x, y, class_name)
        self.cur_class = 'probes'   # toggle with 'b' / 'p'
        self.latest_stamp_ = None
        
        img_sub    = message_filters.Subscriber(self, Image,  '/main_mission/aligned_image')
        aruco_sub  = message_filters.Subscriber(self, String, '/aruco/markers')

        self.sync_ = message_filters.ApproximateTimeSynchronizer(
            [img_sub, aruco_sub], queue_size=10, slop=0.1)
        self.sync_.registerCallback(self.synced_cb)

        # ── publish fake detections in the same format YOLO uses ──────── #
        # pose_estimator_node listens to /detections/raw and expects:
        # {"stamp": {"sec": .., "nanosec": ..},
        #  "detections": [{"class_name": .., "confidence": 1.0,
        #                  "bbox": [x1,y1,x2,y2], "center": [cx,cy]}, ...]}
        self.det_pub_ = self.create_publisher(String, '/detections/raw', 10)

        self.get_logger().info(
            "\n=== Manual Labeler ===\n"
            "  LEFT CLICK  = add point (current class shown in window title)\n"
            "  RIGHT CLICK = undo last click\n"
            "  P           = switch to class 'probes'\n"
            "  B           = switch to class 'rako' (box)\n"
            "  ENTER       = publish detections & advance to next frame\n"
            "  ESC         = skip this frame (no publish)\n"
            "  Q           = quit\n")

    # ── mouse callback ────────────────────────────────────────────────── #
    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicks_.append((x, y, self.cur_class))
        elif event == cv2.EVENT_RBUTTONDOWN and self.clicks_:
            self.clicks_.pop()

    # ── image callback: show frame, collect clicks, publish ───────────── #
    def image_cb(self, img_msg: Image):
        self.latest_stamp_ = img_msg.header.stamp
        frame = self.bridge_.imgmsg_to_cv2(img_msg, 'bgr8')
        self.clicks_.clear()

        win = f"ManualLabeler  |  class: {self.cur_class}  |  ENTER=save  ESC=skip  Q=quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self._mouse_cb)

        while True:
            vis = frame.copy()

            # draw clicks
            for (cx, cy, cls) in self.clicks_:
                color = (0, 255, 0) if cls == 'probes' else (0, 128, 255)
                cv2.circle(vis, (cx, cy), 8, color, -1)
                cv2.putText(vis, cls, (cx + 10, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # update title to show current class
            title = f"ManualLabeler  |  class: {self.cur_class}  |  ENTER=save  ESC=skip  Q=quit"
            cv2.setWindowTitle(win, title)
            cv2.imshow(win, vis)

            key = cv2.waitKey(30) & 0xFF

            if key in (13, 10):          # ENTER → publish & next frame
                cv2.destroyWindow(win)
                self._publish_detections()
                return

            elif key == 27:              # ESC → skip
                cv2.destroyWindow(win)
                return

            elif key == ord('q'):        # Q → quit
                cv2.destroyAllWindows()
                raise SystemExit

            elif key == ord('p'):
                self.cur_class = 'probes'

            elif key == ord('b'):
                self.cur_class = 'rako'

    # ── build and publish /detections/raw in YOLO format ─────────────── #
    def _publish_detections(self):
        if not self.clicks_ or self.latest_stamp_ is None:
            return

        detections = []
        BOX_HALF = 20   # half-size of synthetic bbox around each click point

        for (cx, cy, cls) in self.clicks_:
            detections.append({
                'class_name':  cls,
                'confidence':  1.0,
                'bbox':        [cx - BOX_HALF, cy - BOX_HALF,
                                cx + BOX_HALF, cy + BOX_HALF],
                'center':      [float(cx), float(cy)],
                'image_path':  '',
            })

        payload = {
            'stamp': {
                'sec':     self.latest_stamp_.sec,
                'nanosec': self.latest_stamp_.nanosec,
            },
            'detections': detections,
        }

        self.det_pub_.publish(String(data=json.dumps(payload)))
        self.get_logger().info(
            f"Published {len(detections)} detection(s): "
            + ', '.join(f"{c} @ ({x},{y})" for x, y, c in self.clicks_))


def main(args=None):
    rclpy.init(args=args)
    node = ManualLabeler()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()