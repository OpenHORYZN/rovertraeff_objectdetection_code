import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import config

def ensure_dirs():
    paths = config["paths"]
    for split in ["all", "train", "val", "test"]:
        (paths["images"] / split).mkdir(parents=True, exist_ok=True)
        (paths["labels"] / split).mkdir(parents=True, exist_ok=True)

def clear_dirs():
    paths = config["paths"]
    for split in ["all", "train", "val", "test"]:
        for dir_ in [paths["images"] / split, paths["labels"] / split]:
            if not dir_.exists():
                continue
            for f in dir_.iterdir():
                if f.is_file():
                    f.unlink()

def class_map():
    return {name: i for i, name in enumerate(config["yolo"]["classes"])}

def locate_image(task):
    raw_dir = config["paths"]["raw"]
    rel = task.get("file_upload") or task["data"].get("image", "")
    name = Path(rel).name
    candidate = raw_dir / name
    return candidate if candidate.exists() else None

def convert_box(rect):
    v = rect["value"]
    x = v["x"] / 100.0
    y = v["y"] / 100.0
    w = v["width"] / 100.0
    h = v["height"] / 100.0
    x_c = x + w / 2.0
    y_c = y + h / 2.0
    return x_c, y_c, w, h

def convert_ls_to_yolo():
    paths = config["paths"]
    cmap = class_map()

    with open(paths["ls_export"], "r", encoding="utf-8") as f:
        tasks = json.load(f)

    written_imgs = []

    for task in tasks:
        img_path = locate_image(task)
        if img_path is None:
            print(f"skip: image not found for task {task['id']}")
            continue

        anns = task["annotations"]
        if not anns:
            continue
        results = anns[0].get("result", [])

        yolo_lines = []
        for r in results:
            if r.get("type") != "rectanglelabels":
                continue

            labels = r["value"].get("rectanglelabels", [])
            if not labels:
                continue

            cls_name = labels[0]
            if cls_name not in cmap:
                print(f"unknown class {cls_name} in task {task['id']}")
                continue
            cls_id = cmap[cls_name]

            x_c, y_c, w, h = convert_box(r)
            line = f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
            yolo_lines.append(line)

        # copy image and write label
        all_img = paths["images"] / "all" / img_path.name
        all_lbl = paths["labels"] / "all" / f"{img_path.stem}.txt"

        shutil.copy2(img_path, all_img)
        all_lbl.write_text("\n".join(yolo_lines), encoding="utf-8")

        written_imgs.append(all_img)

    return written_imgs

def split_dataset(imgs):
    paths = config["paths"]
    splits = config["splits"]

    imgs = list(imgs)
    random.seed(splits["seed"])
    random.shuffle(imgs)

    n = len(imgs)
    n_train = int(n * splits["train"])
    n_val = int(n * splits["val"])

    split_map = {
        "train": imgs[:n_train],
        "val": imgs[n_train:n_train + n_val],
        "test": imgs[n_train + n_val:],
    }

    for split, split_imgs in split_map.items():
        for p in split_imgs:
            src_lbl = paths["labels"] / "all" / f"{p.stem}.txt"
            dst_img = paths["images"] / split / p.name
            dst_lbl = paths["labels"] / split / src_lbl.name
            shutil.copy2(p, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

def write_data_yaml():
    paths = config["paths"]
    names = config["yolo"]["classes"]
    name_lines = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    txt = f"""path: {paths["yolo_root"]}
        train: images/train
        val: images/val
        test: images/test

        names:
        {name_lines}
        """
    paths["data_yaml"].write_text(txt, encoding="utf-8")

if __name__ == "__main__":
    ensure_dirs()
    clear_dirs()
    imgs = convert_ls_to_yolo()
    split_dataset(imgs)
    write_data_yaml()
    print(f"converted {len(imgs)} images to YOLO format")