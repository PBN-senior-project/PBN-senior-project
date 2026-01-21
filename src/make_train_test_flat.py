# src/make_train_test_flat.py
from __future__ import annotations

import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import CFG
from src.dataset import _find_image_path  # ใช้ของเดิมคุณ

IMG_COL = "Image Index"
LBL_COL = "Finding Labels"

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path, mode: str = "hardlink"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError("mode must be 'copy' or 'hardlink'")

def is_single_label_only(findings: str, target: str, targets: list[str]) -> bool:
    s = str(findings)
    if target not in s:
        return False
    # NIH multi-label ใช้ | ถ้ามีถือว่าไม่ใช่ single-label
    if "|" in s:
        return False
    for d in targets:
        if d != target and d in s:
            return False
    return True

def choose_paths_for_disease(df_csv: pd.DataFrame, disease: str, cfg: CFG, n: int, used_filenames: set[str]) -> list[Path]:
    sub = df_csv[df_csv[LBL_COL].astype(str).str.contains(disease, regex=False)].copy()

    if getattr(cfg, "ONLY_SINGLE_LABEL", False):
        sub = sub[sub[LBL_COL].apply(lambda x: is_single_label_only(x, disease, cfg.TARGET_DISEASES))]

    sub = sub.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)

    picked: list[Path] = []
    for fname in sub[IMG_COL].astype(str).tolist():
        if fname in used_filenames:
            continue
        p = _find_image_path(cfg.ARCHIVE_DIR, fname)
        if p is None:
            continue
        picked.append(p)
        used_filenames.add(fname)
        if len(picked) >= n:
            break
    return picked

def split_train_test(items: list[tuple[Path, str]], test_split: float, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    n_test = int(round(len(items) * test_split))
    test_idx = set(idx[:n_test].tolist())
    train, test = [], []
    for i, it in enumerate(items):
        if i in test_idx:
            test.append(it)
        else:
            train.append(it)
    return train, test

def main():
    cfg = CFG()

    per_class_total = 2000         # ✅ รวม train+test ต่อโรค
    test_split = getattr(cfg, "TEST_SPLIT", 0.15)
    mode = "hardlink"              # เปลี่ยนเป็น "copy" ถ้า hardlink ใช้ไม่ได้

    out_root = cfg.ROOT / "data" / "processed_flat"
    train_dir = out_root / "train"
    test_dir  = out_root / "test"
    safe_mkdir(train_dir)
    safe_mkdir(test_dir)

    print("[INFO] CSV:", cfg.CSV_PATH)
    print("[INFO] ARCHIVE_DIR:", cfg.ARCHIVE_DIR)
    print("[INFO] TARGET_DISEASES:", cfg.TARGET_DISEASES)
    print("[INFO] ONLY_SINGLE_LABEL:", getattr(cfg, "ONLY_SINGLE_LABEL", False))
    print("[INFO] per_class_total:", per_class_total)
    print("[INFO] test_split:", test_split)
    print("[INFO] output:", out_root)
    print("[INFO] mode:", mode)

    df_csv = pd.read_csv(cfg.CSV_PATH)

    # เก็บไฟล์ที่ใช้ไปแล้ว เพื่อกันรูปเดียวถูกดึงไปซ้ำข้ามโรค (single-label จะสบาย)
    used_filenames: set[str] = set()

    all_items: list[tuple[Path, str]] = []  # (path, label)

    for disease in cfg.TARGET_DISEASES:
        paths = choose_paths_for_disease(df_csv, disease, cfg, per_class_total, used_filenames)
        if len(paths) < per_class_total:
            print(f"[WARN] {disease}: picked {len(paths)} < {per_class_total} (ติด ONLY_SINGLE_LABEL หรือข้อมูลไม่พอ)")
        all_items.extend([(p, disease) for p in paths])

    # shuffle รวมทุกโรคให้ปนกันจริง
    rng = np.random.RandomState(cfg.SEED)
    rng.shuffle(all_items)

    # split train/test จาก “ชุดรวม”
    train_items, test_items = split_train_test(all_items, test_split=test_split, seed=cfg.SEED)

    # กันชื่อไฟล์ชนกัน: ใส่ prefix + running id
    train_rows = []
    for i, (src, label) in enumerate(train_items):
        new_name = f"train_{i:06d}_{label}_{src.name}"
        dst = train_dir / new_name
        link_or_copy(src, dst, mode=mode)
        train_rows.append({"filename": new_name, "label": label})

    test_rows = []
    for i, (src, label) in enumerate(test_items):
        new_name = f"test_{i:06d}_{label}_{src.name}"
        dst = test_dir / new_name
        link_or_copy(src, dst, mode=mode)
        test_rows.append({"filename": new_name, "label": label})

    # save labels mapping
    df_train = pd.DataFrame(train_rows)
    df_test  = pd.DataFrame(test_rows)

    df_train.to_csv(out_root / "train_labels.csv", index=False, encoding="utf-8-sig")
    df_test.to_csv(out_root / "test_labels.csv", index=False, encoding="utf-8-sig")

    # summary
    print("\n[SUMMARY] train per class:")
    print(df_train["label"].value_counts())
    print("\n[SUMMARY] test per class:")
    print(df_test["label"].value_counts())
    print("\n[DONE] saved:", out_root)

if __name__ == "__main__":
    main()
