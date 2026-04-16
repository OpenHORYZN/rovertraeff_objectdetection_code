"""
Benchmark YOLO inference time on your hardware.

Uses settings from config.py. Adjust config and re-run to test different models/resolutions.
"""

import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from config import config


def main():
    # Get config
    yolo_cfg = config["yolo"]
    camera_cfg = config["camera"]
    
    model_name = yolo_cfg["model"]
    device = yolo_cfg["device"]
    conf_threshold = yolo_cfg["confidence"]
    
    width, height = camera_cfg["resolution"]
    
    # Load model
    model_path = Path(__file__).parent / model_name
    print(f"\n{'='*60}")
    print(f"YOLO Benchmark")
    print(f"{'='*60}")
    print(f"Model:      {model_name}")
    print(f"Device:     {device}")
    print(f"Resolution: {width}x{height}")
    print(f"Confidence: {conf_threshold}")
    print(f"Camera:     {camera_cfg['model']}")
    print(f"{'='*60}\n")
    
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    model.to(device)
    
    # Create dummy image (HxWx3)
    dummy_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Warmup
    print("Warming up (5 runs)...")
    for _ in range(5):
        _ = model.predict(dummy_image, conf=conf_threshold, verbose=False)
    
    # Benchmark
    print("Benchmarking (100 runs)...")
    times = []
    for i in range(100):
        start = time.time()
        _ = model.predict(dummy_image, conf=conf_threshold, verbose=False)
        times.append(time.time() - start)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/100")
    
    times = np.array(times) * 1000  # Convert to ms
    
    # Results
    avg_ms = np.mean(times)
    max_fps = 1000 / avg_ms
    
    print(f"\n{'='*60}")
    print(f"Average inference: {avg_ms:.1f} ms → {max_fps:.1f} FPS")
    print(f"Min:               {np.min(times):.1f} ms")
    print(f"Max:               {np.max(times):.1f} ms")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
