# =========================================================
# Build verification pairs CSV (Kaggle-compatible)
# =========================================================

import os, random, itertools
from pathlib import Path
import pandas as pd

# -------- CONFIG --------
DATA_ROOT = "/kaggle/input/cattle-aug-10-muzzle/t_v_10_aug_muzzle"   # your dataset
SPLITS_TO_BUILD = ["val", "test"]  # splits to create pairs for
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
SEED = 42
MAX_POS_PER_CLASS = 10      # cap positives per class
BALANCE_NEG_TO_POS = True   # keep negatives ~= positives
OUT_PREFIX = "pairs"        # files saved as pairs_val.csv / pairs_test.csv
OUTPUT_DIR = Path("/kaggle/working/")  # ⚡ Must be writable in Kaggle
# ------------------------

random.seed(SEED)

def list_class_images(split_dir: Path):
    cls_to_imgs = {}
    for cls_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
        imgs = [str(p) for p in cls_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        if imgs:
            cls_to_imgs[cls_dir.name] = imgs
    all_samples = [(p, cls) for cls, paths in cls_to_imgs.items() for p in paths]
    return cls_to_imgs, all_samples

def build_positive_pairs(cls_to_imgs, max_pos_per_class=MAX_POS_PER_CLASS):
    positives = []
    for cls, paths in cls_to_imgs.items():
        if len(paths) < 2:
            continue
        combos = list(itertools.combinations(paths, 2))
        if max_pos_per_class is not None and len(combos) > max_pos_per_class:
            combos = random.sample(combos, max_pos_per_class)
        for a, b in combos:
            positives.append((a, b, cls))
    return positives

def build_negative_pairs(cls_to_imgs, target_count, seen_pairs):
    negatives = []
    classes = list(cls_to_imgs.keys())
    if len(classes) < 2:
        return negatives
    while len(negatives) < target_count:
        c1, c2 = random.sample(classes, 2)
        im1 = random.choice(cls_to_imgs[c1])
        im2 = random.choice(cls_to_imgs[c2])
        key = tuple(sorted([im1, im2]))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        negatives.append((im1, im2, c1, c2))
    return negatives

def save_pairs_csv(df, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved {len(df)} pairs -> {out_csv}")

def build_pairs_for_split(split_name):
    split_dir = Path(DATA_ROOT)/split_name
    assert split_dir.exists(), f"Split folder not found: {split_dir}"

    cls_to_imgs, all_samples = list_class_images(split_dir)
    print(f"\n=== {split_name.upper()} ===")
    print(f"Classes with images: {len(cls_to_imgs)}  |  Total images: {len(all_samples)}")

    # Positives
    pos = build_positive_pairs(cls_to_imgs, MAX_POS_PER_CLASS)
    random.shuffle(pos)

    # Track seen unordered pairs to avoid duplicates
    seen = set(tuple(sorted([a, b])) for (a, b, _) in pos)

    # Negatives
    neg_target = len(pos) if BALANCE_NEG_TO_POS else len(pos)
    neg = build_negative_pairs(cls_to_imgs, neg_target, seen)

    # Pack to DataFrame
    pos_rows = [{
        "img1": a, "img2": b, "label": 1,
        "class1": c, "class2": c, "split": split_name
    } for (a, b, c) in pos]

    neg_rows = [{
        "img1": a, "img2": b, "label": 0,
        "class1": c1, "class2": c2, "split": split_name
    } for (a, b, c1, c2) in neg]

    df = pd.DataFrame(pos_rows + neg_rows)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # Summary
    n_pos = (df["label"]==1).sum()
    n_neg = (df["label"]==0).sum()
    print(f"Built pairs -> positives: {n_pos} | negatives: {n_neg} | total: {len(df)}")
    print(f"Positives covered classes: {len([k for k,v in cls_to_imgs.items() if len(v)>=2])}")

    # Save CSV to writable path
    out_csv = OUTPUT_DIR/f"{OUT_PREFIX}_{split_name}.csv"
    save_pairs_csv(df, out_csv)

# ------- run -------
for split in SPLITS_TO_BUILD:
    build_pairs_for_split(split)
