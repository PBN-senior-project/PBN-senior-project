import os
import pandas as pd
from typing import Optional


# =====================
# CONFIG (แก้แค่ตรงนี้ถ้าชื่อไฟล์/ที่อยู่ต่าง)
# =====================
CSV_PATH = "Data_Entry_2017.csv"
ARCHIVE_DIR = "archive"
TRAIN_LIST = os.path.join(ARCHIVE_DIR, "train_val_list.txt")
TEST_LIST = os.path.join(ARCHIVE_DIR, "test_list.txt")

IMAGE_ROOT = ARCHIVE_DIR  # มี images_001 ... images_012 อยู่ข้างใน
OUT_DIR = "splits"

IMG_COL = "Image Index"
PATIENT_COL = "Patient ID"
LABEL_COL = "Finding Labels"


def read_list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def find_image_path(image_name: str) -> Optional[str]:
    # Python 3.9 ใช้ Optional[str] แทน (ไม่ใช้ | None)
    for folder in sorted(os.listdir(IMAGE_ROOT)):
        if folder.startswith("images_"):
            p = os.path.join(IMAGE_ROOT, folder, "images", image_name)
            if os.path.exists(p):
                return p
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_LIST):
        raise FileNotFoundError(f"Missing: {TRAIN_LIST}")
    if not os.path.exists(TEST_LIST):
        raise FileNotFoundError(f"Missing: {TEST_LIST}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing: {CSV_PATH}")

    print(" Loading Data_Entry_2017.csv ...")
    df_all = pd.read_csv(CSV_PATH, usecols=[IMG_COL, PATIENT_COL, LABEL_COL])

    print(" Reading list files ...")
    train_names = read_list(TRAIN_LIST)
    test_names = read_list(TEST_LIST)

    print(f" train list = {len(train_names)}")
    print(f" test  list = {len(test_names)}")

    # สร้าง dataframe จากชื่อไฟล์
    train_df = pd.DataFrame({IMG_COL: train_names})
    test_df = pd.DataFrame({IMG_COL: test_names})

    # merge เพื่อดึง metadata
    train_df = train_df.merge(df_all, on=IMG_COL, how="left")
    test_df = test_df.merge(df_all, on=IMG_COL, how="left")

    # เช็ค merge ติดไหม
    train_missing = train_df[train_df[PATIENT_COL].isna() | train_df[LABEL_COL].isna()]
    test_missing = test_df[test_df[PATIENT_COL].isna() | test_df[LABEL_COL].isna()]

    if len(train_missing) > 0:
        raise RuntimeError(
            " train_val_list มีไฟล์ที่ไม่เจอใน CSV (Image Index ไม่ตรง):\n"
            + "\n".join(train_missing[IMG_COL].astype(str).head(30).tolist())
        )
    if len(test_missing) > 0:
        raise RuntimeError(
            " test_list มีไฟล์ที่ไม่เจอใน CSV (Image Index ไม่ตรง):\n"
            + "\n".join(test_missing[IMG_COL].astype(str).head(30).tolist())
        )

    # สร้าง img_path จริง
    print(" Resolving img_path ...")
    train_df["img_path"] = train_df[IMG_COL].apply(find_image_path)
    test_df["img_path"] = test_df[IMG_COL].apply(find_image_path)

    train_noimg = train_df[train_df["img_path"].isna()]
    test_noimg = test_df[test_df["img_path"].isna()]

    if len(train_noimg) > 0:
        raise RuntimeError(
            " train มีรูปที่หาไฟล์ไม่เจอใน archive/images_*/images:\n"
            + "\n".join(train_noimg[IMG_COL].astype(str).head(30).tolist())
        )
    if len(test_noimg) > 0:
        raise RuntimeError(
            " test มีรูปที่หาไฟล์ไม่เจอใน archive/images_*/images:\n"
            + "\n".join(test_noimg[IMG_COL].astype(str).head(30).tolist())
        )

    # (Optional) ถ้าจะทำ single-label: เอา label ตัวแรกก่อน | เช่น "A|B" -> "A"
    train_df[LABEL_COL] = train_df[LABEL_COL].astype(str).apply(lambda x: x.split("|")[0])
    test_df[LABEL_COL] = test_df[LABEL_COL].astype(str).apply(lambda x: x.split("|")[0])

    # save
    train_csv = os.path.join(OUT_DIR, "train.csv")
    test_csv = os.path.join(OUT_DIR, "test.csv")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print("\n🎉 DONE")
    print("Saved:")
    print(" -", train_csv, f"({len(train_df)})")
    print(" -", test_csv, f"({len(test_df)})")

    print("\n🧪 Sample train labels:")
    print(train_df[LABEL_COL].value_counts().head(10))


if __name__ == "__main__":
    main()
