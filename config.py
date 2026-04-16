from pathlib import Path

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

config = {
    "paths": {
        "base": base_dir,
        "data": data_dir,
        "raw": data_dir / "raw" ,                     
        "ls_export": data_dir / "labeled" / "Resultslabeling" / "LSTest" / "test2-norotation-probeonly.json",
        "yolo_root": data_dir / "yolo",
        "images": data_dir / "yolo" / "images",
        "labels": data_dir / "yolo" / "labels",
        "data_yaml": data_dir / "yolo" / "dataset.yaml",
        "runs": data_dir / "runs",
        "vis": data_dir / "yolo" / "vis",
    },
    "splits": {
        "train": 0.7,
        "val": 0.2,
        "test": 0.1,
        "seed": 42,
    },
    "yolo": {
        "model": "yolo11n.pt",  # yolo11s.pt
        "modellive": "data/runs/probe_yolo11_train/weights/best.pt",
        "live_model": "yolo11n.pt",
        "epochs": 50,
        "batch": 4,
        "imgsz": 960,
        "imgszlive": 640,
        "device": "cpu",       # cpu
        "classes": ["probe"], #, "rako_big", "rako_small"
        "live_classes": ["mouse"], # for testing only, delete after
        "confidence": 0.05,    # Detection confidence threshold
        "live_confidence": 0.20,
    },
    "aruco": {
        "enabled": True,
        "dictionary": "DICT_6X6_250", # which one?
        "fallback_dictionaries": ["DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_250"],
        "marker_length": 0.15,
        "reference_marker_id": 101,
    },
    "camera": { 
        "model": "phone", # just for testing
        "device": "http://192.168.178.151:8080/video",  # IP Webcam URL
        "tethering": "http://10.49.75.98:8080/video", # Tethered phone URL
        "resolution": [1920, 1080],
        "fps": 30, 
    },
    "validation": {
        "iou_threshold": 0.5,       # For matching predictions to ground truth
        "conf_threshold": 0.02,     # Low threshold to catch all detections in testing
    },
}