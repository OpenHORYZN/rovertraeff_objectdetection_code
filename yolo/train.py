import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from config import config

def ensure_dirs():
    paths = config["paths"]
    paths["runs"].mkdir(parents=True, exist_ok=True)

def train():
    paths = config["paths"]
    yolo_cfg = config["yolo"]

    model = YOLO(yolo_cfg["model"])

    model.train(
        data=str(paths["data_yaml"]),
        epochs=yolo_cfg["epochs"],
        batch=yolo_cfg["batch"],
        imgsz=yolo_cfg["imgsz"],
        device=yolo_cfg["device"],
        project=str(paths["runs"]),
        name="probe_yolo11_train",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )


if __name__ == "__main__":
    ensure_dirs()
    train()