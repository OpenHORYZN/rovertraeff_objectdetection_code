#!/usr/bin/env python3
"""Minimal live test: phone camera + YOLO + optional ArUco overlay."""

# Stage mapping to ROS nodes:
# - camera capture   -> camera_node
# - YOLO detection   -> yolo_node
# - ArUco detection  -> aruco_node
# - world transform  -> pose_estimator_node (kept in ROS, not in this test script)

import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import config


def build_aruco_detectors(aruco_cfg: dict):
    if not aruco_cfg.get("enabled", False):
        return []

    primary = aruco_cfg.get("dictionary", "DICT_6X6_250")
    fallback = aruco_cfg.get("fallback_dictionaries", [])
    dictionary_names = [primary] + [name for name in fallback if name != primary]

    detectors = []
    for dictionary_name in dictionary_names:
        dictionary_id = getattr(cv2.aruco, dictionary_name, None)
        if dictionary_id is None:
            continue
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        detectors.append((dictionary_name, detector))
    return detectors


def detect_aruco(frame, detectors):
    if not detectors:
        return frame, [], None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for dictionary_name, detector in detectors:
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            continue

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        marker_ids = [int(marker_id) for marker_id in ids.flatten()]
        return frame, marker_ids, dictionary_name

    return frame, [], None


def save_detection(csv_path: Path, image_dir: Path, frame, detection: dict, timestamp: float):
    class_name = detection["class_name"]
    confidence = detection["confidence"]
    x1, y1, x2, y2 = [float(value) for value in detection["bbox"]]

    pad_x = int(0.08 * (x2 - x1))
    pad_y = int(0.08 * (y2 - y1))
    h, w = frame.shape[:2]
    left = max(0, int(x1) - pad_x)
    top = max(0, int(y1) - pad_y)
    right = min(w, int(x2) + pad_x)
    bottom = min(h, int(y2) + pad_y)

    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None

    ts = time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime(timestamp))
    suffix = int((timestamp - int(timestamp)) * 1000)
    image_name = f"{ts}_{suffix:03d}_{class_name}_{confidence:.2f}.jpg"
    image_path = image_dir / image_name
    cv2.imwrite(str(image_path), crop)

    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                f"{timestamp:.6f}",
                class_name,
                f"{confidence:.4f}",
                f"{x1:.2f}",
                f"{y1:.2f}",
                f"{x2:.2f}",
                f"{y2:.2f}",
                f"{center_x:.2f}",
                f"{center_y:.2f}",
                str(image_path),
            ]
        )
    return image_path


def main():
    yolo_cfg = config["yolo"]
    camera_cfg = config["camera"]
    aruco_cfg = config.get("aruco", {})

    camera_url = camera_cfg.get("tethering") or camera_cfg["device"]
    model_cfg = yolo_cfg.get("modellive")
    model_path = Path(__file__).parent / model_cfg
    confidence = float(yolo_cfg.get("live_confidence", yolo_cfg["confidence"]))
    imgsz = int(yolo_cfg.get("imgszlive", yolo_cfg.get("imgsz", 640)))
    desired_classes = yolo_cfg.get("live_classes", yolo_cfg.get("classes", []))

    detections_dir = Path(__file__).parent / "data" / "detections" / "live_phone_yolo"
    image_dir = detections_dir / "images"
    csv_path = detections_dir / "detections.csv"
    image_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "timestamp",
                    "class_name",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "center_x",
                    "center_y",
                    "image_path",
                ]
            )

    print(f"Loading YOLO: {model_path}")
    model = YOLO(str(model_path))
    model.to(yolo_cfg["device"])

    class_ids = None
    if desired_classes:
        name_to_id = {str(v).lower(): int(k) for k, v in model.names.items()}
        class_ids = [name_to_id[name.lower()] for name in desired_classes if name.lower() in name_to_id]

    aruco_detectors = build_aruco_detectors(aruco_cfg)

    print(f"Connecting to camera: {camera_url}")
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  camera_cfg["resolution"][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg["resolution"][1])
    cap.set(cv2.CAP_PROP_FPS,          camera_cfg["fps"])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera stream: {camera_url}")

    print("Running live test. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed. Reconnecting...")
            cap.release()
            time.sleep(0.3)
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        results = model.predict(frame, conf=confidence, imgsz=imgsz, classes=class_ids, verbose=False)
        result = results[0]
        annotated = result.plot()

        # Save one photo entry per detection event in this frame.
        now = time.time()
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = str(model.names.get(class_id, class_id))
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            detection = {
                "class_name": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            }
            save_detection(csv_path, image_dir, frame, detection, now)

        annotated, marker_ids, marker_family = detect_aruco(annotated, aruco_detectors)
        if marker_ids:
            cv2.putText(
                annotated,
                f"ArUco IDs: {marker_ids} [{marker_family}]",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

        cv2.imshow("Live Phone YOLO", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
