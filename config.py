from pathlib import Path

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"
ls_export = data_dir / "labeled" / "export.json"
yolo_root = data_dir / "yolo"

config = {
    "paths": {
        "base": base_dir,
        "data": data_dir,
        "ls_export": ls_export,
        "yolo_root": yolo_root,
        "images": yolo_root / "images",
        "labels": yolo_root / "labels",
        "data_yaml": yolo_root / "dataset.yaml",
    },
    "splits": {
        "train": 0.7,
        "val": 0.2,
        "test": 0.1,
        "seed": 42,
    },
    "yolo": {
        "model": "yolov8n.pt",  # yolo11s.pt
        "epochs": 50,
        "batch": 16,
        "imgsz": 960,
        "device": "cuda",       # cpu
        "classes": ["probe", "rako_big", "rako_small"],
    },
}