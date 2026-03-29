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
        "epochs": 50,
        "batch": 4,
        "imgsz": 960,
        "device": "cpu",       # cpu
        "classes": ["probe"], #, "rako_big", "rako_small"
    },
}