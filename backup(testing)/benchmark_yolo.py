#!/usr/bin/env python3
"""
Offline YOLO benchmark on saved test images.

Compares multiple trained weight files on the same test set and reports
inference time per image and effective FPS.
"""

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import config


def load_model(model_path: Path, device: str):
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    model.to(device)
    return model


def collect_test_images():
    test_img_dir = config["paths"]["yolo_root"] / "test" / "images"
    if not test_img_dir.exists():
        raise FileNotFoundError(f"Test image directory not found: {test_img_dir}")

    image_paths = sorted(
        p for p in test_img_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    )

    if not image_paths:
        raise FileNotFoundError(f"No test images found in: {test_img_dir}")

    return image_paths


def benchmark_model(model, image_paths, yolo_cfg, warmup_images=5):
    device = yolo_cfg["device"]
    conf = float(yolo_cfg["confidence"])
    imgsz = int(yolo_cfg["imgsz"])

    timings_ms = []

    print(f"\nWarmup on {min(warmup_images, len(image_paths))} images")
    for image_path in image_paths[:warmup_images]:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        _ = model.predict(image, conf=conf, imgsz=imgsz, device=device, verbose=False)

    print(f"Benchmarking on {len(image_paths)} images")
    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        start = time.perf_counter()
        _ = model.predict(image, conf=conf, imgsz=imgsz, device=device, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings_ms.append(elapsed_ms)

        if index % 20 == 0:
            print(f"  {index}/{len(image_paths)}")

    if not timings_ms:
        raise RuntimeError("No valid images were benchmarked")

    timings_ms = np.array(timings_ms, dtype=np.float64)

    avg_ms = float(np.mean(timings_ms))
    median_ms = float(np.median(timings_ms))
    p95_ms = float(np.percentile(timings_ms, 95))
    min_ms = float(np.min(timings_ms))
    max_ms = float(np.max(timings_ms))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    return {
        "avg_ms": avg_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "fps": fps,
        "images": len(timings_ms),
    }


def main():
    yolo_cfg = config["yolo"]

    model_specs = [
        ("yolo11n", config["paths"]["runs"] / "FT1TVT1_yolo11_train" / "weights" / "best.pt"),
        ("yolo11s", config["paths"]["runs"] / "FT1TVT1_yolo11s_train_b25" / "weights" / "best.pt"),
        ("yolo11s", config["paths"]["runs"] / "FinalLabel_yolo11" / "weights" / "best.pt"),
    ]

    image_paths = collect_test_images()

    print("\n" + "=" * 72)
    print("OFFLINE YOLO BENCHMARK")
    print("=" * 72)
    print(f"Test images: {len(image_paths)}")
    print(f"Img size:    {yolo_cfg['imgsz']}")
    print(f"Confidence:  {yolo_cfg['confidence']}")
    print(f"Device:      {yolo_cfg['device']}")
    print("=" * 72)

    results = []

    for model_name, model_path in model_specs:
        if not model_path.exists():
            print(f"\nSkipping {model_name}: missing weights at {model_path}")
            continue

        model = load_model(model_path, yolo_cfg["device"])
        metrics = benchmark_model(model, image_paths, yolo_cfg, warmup_images=5)
        metrics["model"] = model_name
        results.append(metrics)

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    for item in results:
        print(
            f"{item['model']}: "
            f"{item['avg_ms']:.1f} ms avg, "
            f"{item['median_ms']:.1f} ms median, "
            f"{item['p95_ms']:.1f} ms p95, "
            f"{item['fps']:.1f} FPS, "
            f"min {item['min_ms']:.1f} ms, "
            f"max {item['max_ms']:.1f} ms"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()