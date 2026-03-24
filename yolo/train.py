from ultralytics import YOLO
from config import config

if __name__ == "__main__":
    ycfg = config["yolo"]
    paths = config["paths"]

    model = YOLO(ycfg["model"])

    model.train(
        data=str(paths["data_yaml"]),
        epochs=ycfg["epochs"],
        device=ycfg["device"],
        batch=ycfg["batch"],
        imgsz=ycfg["imgsz"],
    )