import argparse
from pathlib import Path
from ultralytics import YOLO
from config import config

def infer_images(model, input_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for img in input_dir.glob("*.*"):
        res = model(img)
        res[0].save(filename=str(out_dir / img.name))

def infer_video(model, input_video, out_video):
    res = model(str(input_video), save=True, project=out_video.parent, name=out_video.stem)
    # Ultralytics will handle writing the video; out path is in runs/detect/...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="Folder of images")
    parser.add_argument("--video", type=str, help="Video file")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--weights", type=str, default=None)

    args = parser.parse_args()
    ycfg = config["yolo"]

    weights = args.weights or "runs/detect/train/weights/best.pt"
    model = YOLO(weights)

    out_path = Path(args.output)

    if args.images:
        infer_images(model, Path(args.images), out_path)
    elif args.video:
        infer_video(model, Path(args.video), out_path)
    else:
        print("Specify --images or --video")
