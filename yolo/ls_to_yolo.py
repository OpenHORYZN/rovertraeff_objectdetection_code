import json
import random
import shutil
from pathlib import Path
from config import config

def convert_ls_to_yolo():
    paths = config["paths"]
    classes = config["yolo"]["classes"]

    with open(paths["ls_export"], "r", encoding="utf-8") as f:
        data = json.load(f)

def split_dataset():
    paths = config["paths"]
    splits = config["splits"]

    all_images = sorted((paths["images"] / "all").glob("*.*"))

    random.seed(splits["seed"])
    random.shuffle(all_images)

    n = len(all_images)
    n_train = int(n * splits["train"])
    n_val = int(n * splits["val"])

    split_map = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:],
    }

    for split, imgs in split_map.items():
        (paths["images"] / split).mkdir(parents=True, exist_ok=True)
        (paths["labels"] / split).mkdir(parents=True, exist_ok=True)
        for img_path in imgs:
            label_path = paths["labels"] / "all" / (img_path.stem + ".txt")
            shutil.copy2(img_path, paths["images"] / split / img_path.name)
            shutil.copy2(label_path, paths["labels"] / split / label_path.name)

def write_dataset_yaml():
    paths = config["paths"]
    classes = config["yolo"]["classes"]

    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))

    yaml_text = f"""path: {paths["yolo_root"]}
train: images/train
val: images/val
test: images/test

names:
{names_block}
"""

    paths["data_yaml"].write_text(yaml_text)

if __name__ == "__main__":
    convert_ls_to_yolo()
    split_dataset()
    write_dataset_yaml()        