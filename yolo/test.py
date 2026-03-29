import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from ultralytics import YOLO
from config import config


IOU_THRESH = 0.5
CONF_THRESH = 0.02

COLORS = {
    "gt": (0, 255, 255),   # yellow in BGR
    "tp": (0, 255, 0),     # green
    "fp": (0, 0, 255),     # red
    "fn": (255, 0, 0),     # blue
}


def ensure_dirs():
    config["paths"]["vis"].mkdir(parents=True, exist_ok=True)


def xywhn_to_xyxy(box, img_w, img_h):
    x_c, y_c, w, h = box
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    return [x1, y1, x2, y2]


def compute_iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)

    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0

    return inter_area / union


def load_gt_boxes(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])
            xyxy = xywhn_to_xyxy((x_c, y_c, w, h), img_w, img_h)
            boxes.append({
                "cls_id": cls_id,
                "box": xyxy,
                "matched": False,
            })
    return boxes


def draw_box(img, box, color, label):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        label,
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def get_best_weights():
    weights = config["paths"]["runs"] / "probe_yolo11_train" / "weights" / "best.pt"
    return weights


def test_and_visualize():
    paths = config["paths"]
    yolo_cfg = config["yolo"]

    weights = get_best_weights()
    model = YOLO(str(weights))

    test_img_dir = paths["images"] / "test"
    test_lbl_dir = paths["labels"] / "test"
    vis_dir = paths["vis"]

    image_paths = sorted([
        p for p in test_img_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        label_path = test_lbl_dir / f"{img_path.stem}.txt"
        gt_boxes = load_gt_boxes(label_path, img_w, img_h)

        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESH,
            imgsz=yolo_cfg["imgsz"],
            device=yolo_cfg["device"],
            verbose=False,
        )

        pred_boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                pred_boxes.append({
                    "cls_id": cls_id,
                    "conf": conf,
                    "box": [x1, y1, x2, y2],
                    "matched": False,
                })

        for pred in pred_boxes:
            best_iou = 0.0
            best_gt = None

            for gt in gt_boxes:
                if gt["matched"]:
                    continue
                if pred["cls_id"] != gt["cls_id"]:
                    continue

                iou = compute_iou(pred["box"], gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_gt is not None and best_iou >= IOU_THRESH:
                pred["matched"] = True
                best_gt["matched"] = True

        vis_img = img.copy()

        for gt in gt_boxes:
            draw_box(vis_img, gt["box"], COLORS["gt"], "GT")

        for pred in pred_boxes:
            if pred["matched"]:
                draw_box(vis_img, pred["box"], COLORS["tp"], f"TP {pred['conf']:.2f}")
            else:
                draw_box(vis_img, pred["box"], COLORS["fp"], f"FP {pred['conf']:.2f}")

        for gt in gt_boxes:
            if not gt["matched"]:
                draw_box(vis_img, gt["box"], COLORS["fn"], "FN")

        out_path = vis_dir / img_path.name
        cv2.imwrite(str(out_path), vis_img)

    print(f"saved visualizations to {vis_dir}")


if __name__ == "__main__":
    ensure_dirs()
    test_and_visualize()