import pandas as pd
from pathlib import Path
import shutil
import os
from glob import glob

# =========================
# CONFIG
# =========================
# Notebook อยู่ที่ src/
# Dataset อยู่ที่ data/nih/
BASE_DIR = Path("..")                     # ถอยจาก src -> root
ROOT     = BASE_DIR / "data" / "nih"

CSV_PATH = ROOT / "Data_Entry_2017.csv"
IMG_ROOT = ROOT
OUT_DIR  = ROOT / "No_Finding"

MODE = "copy"        # "copy" หรือ "hardlink"

# =========================
# VALIDATION
# =========================
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

# =========================
# SETUP
# =========================
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("[INFO] Loading CSV...")
df = pd.read_csv(CSV_PATH)

# =========================
# FILTER No Finding
# =========================
if "Finding Labels" not in df.columns or "Image Index" not in df.columns:
    raise ValueError("CSV missing required columns")

df_nf = df[df["Finding Labels"] == "No Finding"].copy()
print(f"[INFO] No Finding samples in CSV: {len(df_nf)}")

# =========================
# BUILD IMAGE PATH MAP
# =========================
print("[INFO] Indexing image paths...")

# รองรับ images_001, images_002, ...
all_images = glob(str(IMG_ROOT / "images_*" / "images" / "*.png"))

img_map = {
    Path(p).name: Path(p)
    for p in all_images
}

print(f"[INFO] Total images indexed: {len(img_map)}")

# =========================
# COPY / LINK FILES
# =========================
missing = 0
processed = 0

for fname in df_nf["Image Index"]:
    src = img_map.get(fname)

    if src is None:
        missing += 1
        continue

    dst = OUT_DIR / fname

    if dst.exists():
        continue

    try:
        if MODE == "copy":
            shutil.copy2(src, dst)
        elif MODE == "hardlink":
            os.link(src, dst)
        else:
            raise ValueError("MODE must be 'copy' or 'hardlink'")
    except Exception as e:
        print(f"[ERROR] {fname}: {e}")
        continue

    processed += 1

# =========================
# SUMMARY
# =========================
print("=" * 40)
print("[DONE] No Finding extraction finished")
print(f"[INFO] Files written: {processed}")
print(f"[WARN] Missing images: {missing}")
print(f"[INFO] Output folder: {OUT_DIR.resolve()}")
print("=" * 40)
