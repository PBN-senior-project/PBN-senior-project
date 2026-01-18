# train_densenet_5cls.py
# Train DenseNet121 (5-class) using archive/train_val_list.txt
# Output: densenet_5cls_best.keras and labels_5cls.txt

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# =====================
# CONFIG
# =====================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
LR = 1e-4

DATA_DIR = "archive"
TRAIN_LIST = os.path.join(DATA_DIR, "train_val_list.txt")
CSV_PATH = "Data_Entry_2017.csv"

MODEL_OUT = "densenet_5cls_best.keras"
LABELS_OUT = "labels_5cls.txt"

# ✅ 5 โรค (ลำดับนี้สำคัญ: ถ้ามีหลายโรค จะเลือกตัวแรก)
CLASSES = ["Atelectasis", "Cardiomegaly", "Edema", "Emphysema", "Fibrosis"]

# =====================
# HELPERS
# =====================
def find_image_path(image_name: str):
    # NIH images อยู่กระจาย images_001..images_012
    for i in range(1, 13):
        p = f"archive/images_{i:03d}/images/{image_name}"
        if os.path.exists(p):
            return p
    return None

def pick_one_label(finding_labels: str):
    """คืน class_name 1 ตัว ถ้ามีใน 5 โรค, ไม่งั้นคืน None"""
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

def make_ds(df, training: bool):
    y = df["y"].astype(np.int32).to_numpy()
    paths = df["img_path"].astype(str).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(2000)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def compute_class_weights(y_int, n_classes: int):
    """กันคลาสไม่บาลานซ์: weight = total/(n_classes*count)"""
    y_int = np.asarray(y_int)
    total = len(y_int)
    weights = {}
    for k in range(n_classes):
        cnt = int((y_int == k).sum())
        weights[k] = (total / (n_classes * max(cnt, 1)))
    return weights

# =====================
# LOAD LIST + CSV
# =====================
with open(TRAIN_LIST, "r", encoding="utf-8") as f:
    train_files = [l.strip() for l in f if l.strip()]

df = pd.read_csv(CSV_PATH)

# เฉพาะภาพที่อยู่ใน train_val_list
df = df[df["Image Index"].isin(train_files)].copy()

# เลือก label 1 ตัวใน 5 โรค
df["class_name"] = df["Finding Labels"].apply(pick_one_label)

# นับ skipped เพราะไม่มี target class
skipped_no_target = int(df["class_name"].isna().sum())

# เหลือเฉพาะที่มี class
df = df.dropna(subset=["class_name"]).copy()

# หา path รูป
df["img_path"] = df["Image Index"].apply(find_image_path)

skipped_missing_img = int(df["img_path"].isna().sum())
df = df.dropna(subset=["img_path"]).copy()

# map class -> id
class_to_id = {c: i for i, c in enumerate(CLASSES)}
df["y"] = df["class_name"].map(class_to_id).astype(int)

print("=== TRAIN (5-class) ===")
print("Total rows in train_list:", len(train_files))
print("Skipped (no target label):", skipped_no_target)
print("Skipped (missing image):", skipped_missing_img)
print("Evaluated samples:", len(df))
print(df["class_name"].value_counts())

# save labels file (ใช้กับ test)
with open(LABELS_OUT, "w", encoding="utf-8") as f:
    for c in CLASSES:
        f.write(c + "\n")
print("✅ Saved:", LABELS_OUT)

# =====================
# SPLIT train/val
# =====================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["y"]
)

train_ds = make_ds(train_df, training=True)
val_ds = make_ds(val_df, training=False)

# =====================
# MODEL (DenseNet121 5-class)
# =====================
base = tf.keras.applications.DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
out = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)

model = tf.keras.Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# class weights (ช่วยแก้ imbalance)
class_weights = compute_class_weights(train_df["y"].values, len(CLASSES))
print("Class weights:", class_weights)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    ),
]

# =====================
# TRAIN
# =====================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# save final (checkpoint มีแล้ว แต่อันนี้กันพลาด)
model.save(MODEL_OUT)
print(" Saved model:", MODEL_OUT)
