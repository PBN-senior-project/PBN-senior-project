import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, ResNet50V2, MobileNetV2
from tensorflow.keras import layers, models, optimizers, callbacks
import os
import numpy as np
from sklearn.metrics import roc_auc_score

# ================= ⚙️ V7 SETUP: Mixed Precision =================
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
# ================================================================

# ================= ⚙️ CONFIG V7 - 6 DISEASES =================

CSV_PATH = "/app/archive/Data_Entry_2017.csv"

# ARCHIVE_DIR = r"D:\senior-project\ploy-senior-project\archive"
ARCHIVE_DIR = "/app/archive"

MODEL_SAVE_DIR = "/app/models_v7"
GRAPH_SAVE_DIR = "/app/outputs/graphs_v7"

IMG_SIZE = (384, 384)
BATCH_SIZE = 8
EPOCHS = 10 #40
INITIAL_LR = 1e-4

CLASSES = [
    'Infiltration',
    'Effusion',
    'Atelectasis',
    'Nodule',
    'Mass',
    'Pneumothorax'
]

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(GRAPH_SAVE_DIR, exist_ok=True)
# ==================================================

# ---------------------------------------------------------
# 1️⃣ Load Data
# ---------------------------------------------------------
print("⏳ Loading Data...")
df = pd.read_csv(CSV_PATH)

# ================================
# Create Multi-label Columns
# ================================
# for cls in CLASSES:
#     df[cls] = df["Finding Labels"].apply(
#         lambda x: 1 if cls in str(x).split("|") else 0
#     )

# train_df = df.sample(frac=0.8, random_state=42)
# val_df = df.drop(train_df.index)

import glob

# ==================================================
# Find all image files inside archive
# ==================================================
image_paths = glob.glob(
    os.path.join(ARCHIVE_DIR, "**", "*.png"),
    recursive=True
)

image_map = {
    os.path.basename(path): path
    for path in image_paths
}

df["filepath"] = df["Image Index"].map(image_map)

print(f"📁 Images found in folders: {len(image_paths)}")
print(f"✅ Matched with CSV: {df['filepath'].notna().sum()}")
print(f"❌ Missing images: {df['filepath'].isna().sum()}")

# เอาเฉพาะแถวที่มีไฟล์ภาพจริง
df = df[df["filepath"].notna()].copy()
for cls in CLASSES:
    df[cls] = df["Finding Labels"].apply(
        lambda x: 1 if cls in str(x).split("|") else 0
    )

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

# ---------------------------------------------------------
# ⭐ [V7 UPDATE] 1.5 Targeted Minority Oversampling
# แก้ปัญหา Imbalance ที่ระดับ Data (จาก Irtaza 2024)
# ---------------------------------------------------------
print("\n Applying Targeted Oversampling for Minority Classes...")
class_counts = train_df[CLASSES].sum()
# หาโรคหายากที่มีตัวอย่างน้อยกว่า 2000 รูปใน Training set
minority_classes = class_counts[class_counts < 2000].index.tolist()
print(f"   🎯 Minority Classes to Oversample: {minority_classes}")

# ดึงแถวที่มีโรคหายากออกมา
minority_df = train_df[train_df[minority_classes].sum(axis=1) > 0]
# ทำการ Copy ข้อมูลกลุ่มนี้เพิ่ม 1 เท่าตัว (Oversampling)
train_df_balanced = pd.concat([train_df, minority_df], ignore_index=True)
# สลับข้อมูลให้กระจายตัว
train_df = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"   📈 Train data size increased from {len(df)*0.8:.0f} to {len(train_df)} images.\n")

# ---------------------------------------------------------
# 2️⃣ SOTA Algorithm: Weighted Focal Loss
# ---------------------------------------------------------
pos_counts = train_df[CLASSES].sum().values 
total = len(train_df)
# คำนวณ Weights จาก Dataset ที่ปรับสมดุลแล้ว
weight_values = (total / (pos_counts + 1e-5)) / 2.0 
pos_weights_tensor = tf.constant(weight_values, dtype=tf.float32)

def weighted_focal_loss(pos_weights, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_factor = y_true * pos_weights + (1 - y_true) * 1.0 

        loss = -alpha_factor * modulating_factor * tf.math.log(p_t)
        return tf.reduce_mean(loss)
    return loss_fn

# ---------------------------------------------------------
# ⭐ [V7 UPDATE] 3️⃣ Generators & Medical Augmentation
# ---------------------------------------------------------
import cv2  # ⭐ [V7 UPDATE - เพิ่มใหม่] ใช้สำหรับทำ CLAHE (Adaptive Windowing)

# ==========================================================
# ⭐ [V7 UPDATE - เพิ่มใหม่ทั้งฟังก์ชัน] Custom Preprocessing
# รวม 3 เทคนิค: CLAHE + Gamma Correction + Gaussian Noise
# ทำงานหลัง rescale (ภาพเป็น float32 ช่วง 0-1) จึงต้อง denormalize
# ก่อนใช้ OpenCV แล้วค่อย normalize กลับ
# ==========================================================
def medical_preprocessing(img):
    # ImageDataGenerator ส่งภาพเข้าฟังก์ชันนี้ก่อน rescale
    # ดังนั้น input ยังอยู่ช่วงประมาณ 0-255
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)

    # ⭐ เทคนิคที่ 1: CLAHE
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_uint8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # ⭐ เทคนิคที่ 2: Safe Random Cropping
    # Crop เบา ๆ 384x384 -> 368x368 แล้ว resize กลับ
    h, w = img_uint8.shape[:2]
    crop_h = min(368, h)
    crop_w = min(368, w)

    if h > crop_h:
        top = np.random.randint(0, h - crop_h + 1)
    else:
        top = 0

    if w > crop_w:
        left = np.random.randint(0, w - crop_w + 1)
    else:
        left = 0

    img_uint8 = img_uint8[
        top:top + crop_h,
        left:left + crop_w
    ]

    img_uint8 = cv2.resize(
        img_uint8,
        IMG_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    # Normalize เป็น 0-1
    img = img_uint8.astype(np.float32) / 255.0

    # ⭐ เทคนิคที่ 3: Gamma Correction
    gamma = np.random.uniform(0.85, 1.15)
    img = np.power(img, gamma)

    # ⭐ เทคนิคที่ 4: Gaussian Noise
    if np.random.rand() < 0.5:
        noise = np.random.normal(
            0,
            0.01,
            img.shape
        ).astype(np.float32)
        img = img + noise

    img = np.clip(img, 0.0, 1.0)
    return img


# เอา Brightness ออก และลดการขยับลงครึ่งหนึ่งให้พอดีกับสรีระปอด
train_datagen = ImageDataGenerator(
    rotation_range=10,        # ลดจาก 15
    width_shift_range=0.05,   # ลดจาก 0.1
    height_shift_range=0.05,  # ลดจาก 0.1
    zoom_range=0.1,           # ลดจาก 0.15
    shear_range=5,             # ⭐ [V7 UPDATE - เพิ่มใหม่]
                                # เทคนิค: Shear (บิดภาพตามแนวทแยงเล็กน้อย)
                                # ผลที่ได้: จำลองมุมถ่ายที่เอียงเล็กน้อยจาก
                                # ท่ายืน/นั่งของผู้ป่วยตอนถ่ายฟิล์ม ใส่แค่ 5 องศา (น้อยกว่าค่า default ทั่วไป) เพราะปอด
                                # มีโครงสร้างตายตัว บิดมากไปจะผิดกายวิภาค
    fill_mode='constant', cval=0,
    horizontal_flip=True,      # คงไว้ตามเปเปอร์ CheXNet (Stanford)
    preprocessing_function=medical_preprocessing
    # ⭐ [V7 UPDATE - เพิ่มใหม่] เรียกใช้ฟังก์ชันด้านบน
    # ผลรวม: เพิ่ม CLAHE + Gamma + Gaussian Noise เข้าไปใน pipeline
    # ของ train_datagen ทำงานหลัง rescale ทุกครั้งที่สุ่มภาพมา train
)

# ⭐ [V7 UPDATE - เพิ่มใหม่ทั้งฟังก์ชัน] Preprocessing สำหรับ validation/TTA
# ใช้ CLAHE เหมือนกัน (deterministic) แต่ "ไม่" ใส่ gamma/noise แบบสุ่ม
# เหตุผล: ตอน validate ต้องการผลลัพธ์ที่ reproducible ทุกครั้ง ไม่อยาก
# ให้ metric แกว่งเพราะ noise สุ่ม แต่ต้องการให้ contrast ของภาพ (CLAHE)
# สอดคล้องกับที่โมเดลเห็นตอน train เพื่อไม่ให้เกิด distribution mismatch
def val_preprocessing(img):
    # Validation ใช้ CLAHE เหมือน train แต่ไม่มี Random Crop / Gamma / Noise
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_uint8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img_uint8.astype(np.float32) / 255.0

val_datagen = ImageDataGenerator(
    preprocessing_function=val_preprocessing
    # ⭐ [V7 UPDATE - เพิ่มใหม่] เดิมมีแค่ rescale=1./255 บรรทัดเดียว
    # ตอนนี้เพิ่ม CLAHE เข้าไปให้สอดคล้องกับ train_datagen
)

# IMG_DIR = ARCHIVE_DIR

# train_generator = train_datagen.flow_from_dataframe(
#     dataframe=train_df, directory=IMG_DIR, x_col='Filename', y_col=CLASSES,
#     target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='raw', shuffle=True
# )
# val_generator = val_datagen.flow_from_dataframe(
#     dataframe=val_df, directory=IMG_DIR, x_col='Filename', y_col=CLASSES,
#     target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='raw', shuffle=False
# )
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="filepath",
    y_col=CLASSES,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=None,
    x_col="filepath",
    y_col=CLASSES,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False
)

# ==========================================================
# ⭐ [V7 UPDATE - เพิ่มใหม่ทั้ง block] Test-Time Augmentation (TTA)
# เทคนิค: TTA (สร้าง generator ที่ 2 ของชุด validation เดิม 
# แต่เพิ่มการ Shift และ Flip เข้าไปแบบสุ่ม)
# ผลที่ได้: ตอน evaluation/ensemble (section 7) จะ predict ภาพหลายเวอร์ชัน
# (ภาพจริง + ภาพที่ผ่าน shift/flip) แล้วเอาผลมาเฉลี