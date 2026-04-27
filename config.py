from pathlib import Path

base_dir = Path(__file__).resolve().parents[5]
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
        "device": "cpu",
        "epochs": 50,
        "batch": 4,
        "imgsz": 640, # phone: 640, d455: 960   
        "classes": ["probe"], # "probe", "rako_big", "rako_small", testing: mouse
        "confidence": 0.03,    # Detection confidence threshold, live: 0.20
    },
    "aruco": {
        "enabled": True,
        "dictionary": "DICT_5X5_250", # 250?
        "fallback_dictionaries": ["DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000"],
        "marker_length": 0.15,
        "reference_marker_id": 101,
    },
    "pose_estimator": {
        "log_points": True,
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