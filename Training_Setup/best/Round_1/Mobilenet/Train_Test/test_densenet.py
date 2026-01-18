# test_densenet_5cls.py
# Evaluate DenseNet (5-class) using archive/test_list.txt
# Prints classification_report + confusion matrix like your Mobilenet output

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# =====================
# CONFIG
# =====================
IMG_SIZE = 224
BATCH_SIZE = 32

DATA_DIR = "archive"
TEST_LIST = os.path.join(DATA_DIR, "test_list.txt")
CSV_PATH = "Data_Entry_2017.csv"

MODEL_PATH = "densenet_5cls_best.keras"
LABELS_PATH = "labels_5cls.txt"

# =====================
# LOAD LABELS
# =====================
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    CLASSES = [line.strip() for line in f if line.strip()]

class_to_id = {c: i for i, c in enumerate(CLASSES)}

# =====================
# HELPERS
# =====================
def find_image_path(image_name: str):
    for i in range(1, 13):
        p = f"archive/images_{i:03d}/images/{image_name}"
        if os.path.exists(p):
            return p
    return None

def pick_one_label(finding_labels: str):
    s = str(finding_labels)
    for c in CLASSES:
        if c in s:
            return c
    return None

def load_image(path, y):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img, y

# =====================
# LOAD MODEL
# =====================
model = tf.keras.models.load_model(MODEL_PATH)

# =====================
# LOAD TEST LIST + CSV
# =====================
with open(TEST_LIST, "r", encoding="utf-8") as f:
    test_files = [l.strip() for l in f if l.strip()]

df = pd.read_csv(CSV_PATH)
df = df[df["Image Index"].isin(test_files)].copy()

df["class_name"] = df["Finding Labels"].apply(pick_one_label)
skipped_no_target = int(df["class_name"].isna().sum())

df = df.dropna(subset=["class_name"]).copy()
df["img_path"] = df["Image Index"].apply(find_image_path)
skipped_missing_img = int(df["img_path"].isna().sum())
df = df.dropna(subset=["img_path"]).copy()

df["y"] = df["class_name"].map(class_to_id).astype(int)

print("Evaluated samples:", len(df))
print("Skipped rows (missing image or no target label):", skipped_no_target + skipped_missing_img)
print()

# =====================
# DATASET
# =====================
paths = df["img_path"].astype(str).to_numpy()
y_true = df["y"].astype(int).to_numpy()

ds = tf.data.Dataset.from_tensor_slices((paths, y_true))
ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =====================
# PREDICT
# =====================
probs = model.predict(ds, verbose=0)          # (N,5)
y_pred = np.argmax(probs, axis=1)

# =====================
# REPORT (เหมือนรูปสอง)
# =====================
print(" " * 5 + "precision    recall  f1-score   support")
print()
print(classification_report(
    y_true,
    y_pred,
    target_names=CLASSES,
    digits=2
))

# =====================
# CONFUSION MATRIX (เหมือนรูปสอง)
# =====================
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))

print("====================================================")
print("Confusion Matrix (แนวตั้ง: ความจริง, แนวนอน: AI ทาย)")
print("====================================================")

# header
header = " " * 12 + " ".join([f"{c:>12s}" for c in CLASSES])
print(header)

for i, row in enumerate(cm):
    row_str = " ".join([f"{v:12d}" for v in row])
    print(f"{CLASSES[i]:<12s}{row_str}")
