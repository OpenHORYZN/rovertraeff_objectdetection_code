from pathlib import Path

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

config = {
    "paths": {
        "base": base_dir,
        "data": data_dir,
        "raw": data_dir / "raw" ,                     
        "ls_export": data_dir / "labeled" / "Resultslabeling" / "LSFlighttest1_YoloOBBImag" / "project-3-at-2026-04-30-14-22-7fea1c75.json",
        "yolo_root": data_dir / "labeled" / "Resultslabeling" / "RbflwFT1TVT",
        "images": data_dir / "yolo" / "images",
        "labels": data_dir / "yolo" / "labels",
        "data_yaml": data_dir / "labeled" / "Resultslabeling" / "RbflwFT1TVT" / "data.yaml",
        "runs": data_dir / "runs",
        "vis": data_dir / "labeled" / "Resultslabeling" / "RbflwFT1TVT" / "vis",
    },
    "splits": {
        "train": 0.7,
        "val": 0.2,
        "test": 0.1,
        "seed": 42,
    },
    "yolo": {
        "model": "yolo11n.pt",  # yolo11s.pt
        "modellive": "data/runs/FT1TVT1_yolo11_train/weights/best.pt",
        "device": "cpu",
        "epochs": 25,
        "batch": 4,
        "imgsz": 960, # phone: 640, d455: 960   
        "classes": ["probe", "rako"], # "probe", "rako_big", "rako_small", testing: mouse
        "confidence": 0.2,    # Detection confidence threshold, live: 0.20
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
        "model": "d455", # just for testing
        "device": "http://192.168.178.151:8080/video",  # IP Webcam URL
        "tethering": "http://10.49.75.98:8080/video", # Tethered phone URL
        "resolution": [1280, 720],
        "fps": 30, 
    },
    "validation": {
        "iou_threshold": 0.5,       # For matching predictions to ground truth
        "conf_threshold": 0.2,     # Low threshold to catch all detections in testing
    },
}