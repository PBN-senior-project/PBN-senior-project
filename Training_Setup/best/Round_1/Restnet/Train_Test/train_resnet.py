# train_resnet.py
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# =====================
# CONFIG (แก้ได้ตรงนี้)
# =====================
SPLIT_DIR = "splits"
MODEL_DIR = "models"

TRAIN_CSV = os.path.join(SPLIT_DIR, "train.csv")

PATH_COL = "img_path"
LABEL_COL = "Finding Labels"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

# เลือกคลาสที่จะเทรน
# - ถ้า TARGET_LABELS เป็น list -> เทรนเฉพาะคลาสนั้น
# - ถ้า TARGET_LABELS = None -> เลือก top K จาก train.csv อัตโนมัติ
TARGET_LABELS = None
TOP_K = 5

# ถ้าไม่อยากให้ "No Finding" อยู่ในคลาส ให้ตั้ง False
INCLUDE_NO_FINDING = True

# สัดส่วน val
VAL_RATIO = 0.2

# learning rate
LR = 1e-4

# เปิด/ปิด augmentation เบา ๆ
USE_AUG = True

# ✅ เพิ่ม: ไฟล์ history สำหรับกราฟ
HISTORY_CSV = "history_resnet.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================
# Utilities
# =====================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def first_label(label_str: str) -> str:
    """ถ้า label เป็น 'A|B|C' จะเอา 'A'"""
    if label_str is None:
        return ""
    s = str(label_str).strip()
    if not s:
        return ""
    return s.split("|")[0].strip()

def load_dataframe(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing: {csv_path}")
    df = pd.read_csv(csv_path)
    if PATH_COL not in df.columns:
        raise ValueError(f"CSV missing column: {PATH_COL}")
    if LABEL_COL not in df.columns:
        raise ValueError(f"CSV missing column: {LABEL_COL}")

    df = df.copy()
    df[PATH_COL] = df[PATH_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(str).apply(first_label)

    # ตัดแถวที่ path ไม่มีไฟล์จริง
    df = df[df[PATH_COL].apply(os.path.exists)].copy()
    df = df[df[LABEL_COL].str.len() > 0].copy()

    return df

def choose_labels(df: pd.DataFrame):
    """เลือก labels ที่จะเทรน"""
    if TARGET_LABELS is not None and len(TARGET_LABELS) > 0:
        labels = list(TARGET_LABELS)
    else:
        vc = df[LABEL_COL].value_counts()
        if not INCLUDE_NO_FINDING and "No Finding" in vc.index:
            vc = vc.drop("No Finding")
        labels = vc.head(TOP_K).index.tolist()

    if len(labels) < 2:
        raise RuntimeError(f"Labels too few: {labels} (ต้องมีอย่างน้อย 2 คลาส)")
    return labels

def save_labels(labels, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for x in labels:
            f.write(str(x).strip() + "\n")

# =====================
# TF Dataset
# =====================
from tensorflow.keras.applications.resnet50 import preprocess_input

def decode_and_preprocess(path: tf.Tensor, label: tf.Tensor):
    """อ่านรูป -> resize -> preprocess ตาม ResNet50"""
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)  # 0..255
    img = preprocess_input(img)     # ResNet50 preprocess
    return img, label

def augment(img: tf.Tensor, label: tf.Tensor):
    if USE_AUG:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.05)
        img = tf.image.random_contrast(img, lower=0.95, upper=1.05)
    return img, label

def make_dataset(paths, labels, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 5000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# =====================
# Model
# =====================
def build_model(num_classes: int):
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling=None
    )
    base.trainable = False  # เริ่มจาก freeze ก่อน

    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model, base

# =====================
# Main Train
# =====================
def main():
    set_seed(SEED)

    print("[INFO] Loading train.csv ...")
    df = load_dataframe(TRAIN_CSV)
    print("[INFO] Rows after clean =", len(df))

    labels = choose_labels(df)
    print("[INFO] Using labels:", labels)

    # filter เฉพาะคลาสที่เลือก
    df = df[df[LABEL_COL].isin(labels)].copy()
    print("[INFO] Rows after label filter =", len(df))

    # label -> index
    label2idx = {name: i for i, name in enumerate(labels)}
    df["y"] = df[LABEL_COL].map(label2idx).astype(int)

    # split train/val แบบ stratify
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=df["y"]
    )

    print("[INFO] Train =", len(train_df), "Val =", len(val_df))

    train_paths = train_df[PATH_COL].values
    train_y = train_df["y"].values
    val_paths = val_df[PATH_COL].values
    val_y = val_df["y"].values

    train_ds = make_dataset(train_paths, train_y, training=True)
    val_ds = make_dataset(val_paths, val_y, training=False)

    model, base = build_model(num_classes=len(labels))
    model.summary()

    # callbacks
    best_path = os.path.join(MODEL_DIR, "resnet_best.keras")
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        best_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        mode="max",
        verbose=1
    )
    reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    # ✅ เพิ่ม CSVLogger (รอบแรก: สร้างไฟล์ใหม่)
    csvlog1 = tf.keras.callbacks.CSVLogger(HISTORY_CSV, append=False)

    print("\n[INFO] Training (frozen backbone) ...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[ckpt, early, reduce, csvlog1],  # ✅ ใส่เพิ่มตรงนี้
        verbose=1
    )

    # (optional) fine-tune เล็กน้อย: unfreeze last blocks
    print("\n[INFO] Fine-tune last layers ...")
    base.trainable = True

    # freeze ช่วงต้น ๆ เพื่อไม่ให้พังเร็ว
    for layer in base.layers[:120]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR * 0.1),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # ✅ รอบ fine-tune: append=True (ต่อท้ายในไฟล์เดิม)
    csvlog2 = tf.keras.callbacks.CSVLogger(HISTORY_CSV, append=True)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max(3, EPOCHS // 2),
        callbacks=[ckpt, early, reduce, csvlog2],  # ✅ ใส่เพิ่มตรงนี้
        verbose=1
    )

    # save labels
    labels_path = os.path.join(MODEL_DIR, "labels.txt")
    save_labels(labels, labels_path)

    print("\n✅ DONE")
    print("Saved model:", best_path)
    print("Saved labels:", labels_path)
    print("Saved history:", HISTORY_CSV)
    print("Labels:", labels)

if __name__ == "__main__":
    main()
