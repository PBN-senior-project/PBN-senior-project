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

ARCHIVE_DIR = r"D:\senior-project\ploy-senior-project\archive"

MODEL_SAVE_DIR = "/app/models_v7"
GRAPH_SAVE_DIR = "/app/outputs/graphs_v7"

IMG_SIZE = (384, 384)
BATCH_SIZE = 8
EPOCHS = 40
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
    # ----- แปลงกลับเป็น uint8 (0-255) เพื่อใช้กับ OpenCV -----
    img_uint8 = (img * 255).astype(np.uint8)

    # ⭐ เทคนิคที่ 1: CLAHE (Adaptive Windowing)
    # ผลที่ได้: ดึง contrast ในเนื้อปอด (บริเวณมืด) ให้เห็นรอยโรคชัดขึ้น
    # ทำไมต้องมี: ภาพ X-ray จาก dataset NIH มาจากหลายเครื่อง/หลาย
    # โรงพยาบาล contrast ไม่เท่ากัน CLAHE ช่วยให้ทุกภาพมี contrast
    # ที่สม่ำเสมอมากขึ้น -> โมเดล generalize ข้ามเครื่อง/โรงพยาบาลได้ดีขึ้น
    # ทำแบบ deterministic (ทำทุกภาพเหมือนกัน ไม่สุ่ม)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_uint8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # ----- กลับเป็น float32 (0-1) -----
    img = img_uint8.astype(np.float32) / 255.0

    # ⭐ เทคนิคที่ 2: Gamma Correction (สุ่มค่า 0.85-1.15 ทุกครั้งที่เรียก)
    # ผลที่ได้: จำลองความสว่าง/ความเข้มของภาพที่ต่างกันไปในแต่ละครั้ง
    # (คล้าย Brightness แต่ non-linear และควบคุมได้ปลอดภัยกว่า)
    # ทำไมต้องมี: ทดแทน Brightness ที่ถูกถอดออกไป โดยเปลี่ยนความสว่าง
    # แบบ "นุ่มนวลกว่า" ไม่ทำให้ค่า pixel หลุดขอบ (clip) ง่ายเท่า brightness ตรงๆ
    gamma = np.random.uniform(0.85, 1.15)
    img = np.power(img, gamma)

    # ⭐ เทคนิคที่ 3: Gaussian Noise (สุ่มใส่ 50% ของภาพ, intensity ต่ำมาก)
    # ผลที่ได้: จำลอง sensor noise หรือ noise จากการบีบอัดไฟล์ภาพ (JPEG)
    # ทำไมต้องมี: ทำให้โมเดล robust ขึ้น ไม่ overfit กับภาพที่ "สะอาด"เกินไป
    # ใส่เบามาก (std=0.01) เพราะถ้าใส่แรงจะกลบรอยโรคเล็กๆ อย่าง Nodule ได้
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 0.01, img.shape).astype(np.float32)
        img = img + noise

    # Clip ให้อยู่ในช่วง 0-1 เสมอ (กันค่าหลุดขอบจาก gamma/noise ด้านบน)
    img = np.clip(img, 0.0, 1.0)
    return img


# เอา Brightness ออก และลดการขยับลงครึ่งหนึ่งให้พอดีกับสรีระปอด
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,        # ลดจาก 15
    width_shift_range=0.05,   # ลดจาก 0.1
    height_shift_range=0.05,  # ลดจาก 0.1
    zoom_range=0.1,           # ลดจาก 0.15
    shear_range=5,             # ⭐ [V7 UPDATE - เพิ่มใหม่]
                                # เทคนิค: Shear (บิดภาพตามแนวทแยงเล็กน้อย)
                                # ผลที่ได้: จำลองมุมถ่ายที่เอียงเล็กน้อยจาก
                                # ท่ายืน/นั่งของผู้ป่วยตอนถ่ายฟิล์ม ใส่แค่ 5
                                # องศา (น้อยกว่าค่า default ทั่วไป) เพราะปอด
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
    img_uint8 = (img * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_uint8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img_uint8.astype(np.float32) / 255.0

val_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=val_preprocessing
    # ⭐ [V7 UPDATE - เพิ่มใหม่] เดิมมีแค่ rescale=1./255 บรรทัดเดียว
    # ตอนนี้เพิ่ม CLAHE เข้าไปให้สอดคล้องกับ train_datagen
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df, directory=IMG_DIR, x_col='Filename', y_col=CLASSES,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='raw', shuffle=True
)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df, directory=IMG_DIR, x_col='Filename', y_col=CLASSES,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='raw', shuffle=False
)

# ==========================================================
# ⭐ [V7 UPDATE - เพิ่มใหม่ทั้ง block] Test-Time Augmentation (TTA)
# เทคนิค: TTA (สร้าง generator ที่ 2 ของชุด validation เดิม แต่ flip แนวนอน)
# ผลที่ได้: ตอน evaluation/ensemble (section 7) จะ predict ภาพ 2 แบบ(ภาพจริง + ภาพ flip) 
# แล้วเอาผลมาเฉลี่ยกัน ช่วยให้ prediction เสถียรขึ้น มักทำให้ AUC ขยับขึ้นเล็กน้อยโดยไม่ต้อง retrain โมเดลใหม่เลย
# ใช้เฉพาะตอน inference เท่านั้น ไม่เกี่ยวกับ training loop
# ==========================================================
tta_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    preprocessing_function=val_preprocessing
)
tta_generator = tta_datagen.flow_from_dataframe(
    dataframe=val_df, directory=IMG_DIR, x_col='Filename', y_col=CLASSES,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='raw', shuffle=False
)

# ---------------------------------------------------------
# 4️⃣ Model Builder
# ---------------------------------------------------------
def build_model(model_name):
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    if model_name == 'DenseNet121':
        base = DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
    elif model_name == 'ResNet50V2':
        base = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
    elif model_name == 'MobileNetV2':
        base = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)

    base.trainable = True
    for layer in base.layers[:-120]: 
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(CLASSES), activation='sigmoid', dtype='float32')(x)

    model = models.Model(inputs=base.input, outputs=outputs, name=model_name)
    
    # [V7 UPDATE] ใช้ AdamW ด้วย LR คงที่เพื่อใช้ร่วมกับ ReduceLROnPlateau
    optimizer = optimizers.AdamW(learning_rate=INITIAL_LR, weight_decay=1e-4)

    model.compile(optimizer=optimizer,
                  loss=weighted_focal_loss(pos_weights_tensor, gamma=2.0),
                  metrics=[tf.keras.metrics.AUC(multi_label=True, name='auc')])
    return model

# ---------------------------------------------------------
# 🚀 5️⃣ MAIN TRAINING LOOP & HISTORY TRACKING
# ---------------------------------------------------------
MODELS_TO_TRAIN = ['DenseNet121', 'ResNet50V2', 'MobileNetV2']
saved_model_paths = []
history_dict = {} 

print("\n" + "="*50)
print("🚀 STARTING V7 FINAL THESIS TRAINING")
print("="*50)

for m_name in MODELS_TO_TRAIN:
    print(f"\n🧠 Training Base Model: {m_name}")
    model = build_model(m_name)
    
    save_path = os.path.join(MODEL_SAVE_DIR, f'best_{m_name}_v7.keras')
    saved_model_paths.append(save_path)
    
    # ⭐ [V7 UPDATE] ใช้ ReduceLROnPlateau ตามเปเปอร์ CheXNet
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.1, patience=2, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(save_path, monitor='val_auc', save_best_only=True, mode='max', verbose=1)
    ]
    
    history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator, callbacks=callbacks_list)
    history_dict[m_name] = history.history['val_auc']
    
    del model
    tf.keras.backend.clear_session()
    print(f"✅ Finished & Saved: {m_name}. VRAM Cleared.")

# ---------------------------------------------------------
# 📈 6️⃣ PLOT TRAINING CURVES
# ---------------------------------------------------------
print("\n Generating Validation AUC Comparison Graph...")
plt.figure(figsize=(12, 8))
for m_name, val_auc_scores in history_dict.items():
    plt.plot(val_auc_scores, label=f'{m_name}', linewidth=2.5, marker='o')

plt.title('V7 Validation AUC Comparison (14 Diseases)', fontsize=16, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Validation AUC Score', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

graph_path = os.path.join(GRAPH_SAVE_DIR, 'training_comparison_v7.png')
plt.savefig(graph_path, bbox_inches='tight', dpi=300)
print(f"✅ Graph saved to: {graph_path}")
plt.close()

# # ---------------------------------------------------------
# # 👑 7️⃣ THE 4TH MODEL: ENSEMBLE EVALUATION
# # ---------------------------------------------------------
# print("\n" + "🌟"*25)
# print("👑 EVALUATING THE ULTIMATE ENSEMBLE MODEL (V7)")
# print("🌟"*25)

# y_true = val_generator.labels[:val_generator.samples]
# ensemble_preds = np.zeros((val_generator.samples, len(CLASSES)))

# for path in saved_model_paths:
#     print(f"⏳ Running Inference with: {os.path.basename(path)}")
#     model = tf.keras.models.load_model(path, compile=False)
#     preds = model.predict(val_generator, steps=len(val_generator), verbose=1)
#     ensemble_preds += preds[:val_generator.samples]
#     del model
#     tf.keras.backend.clear_session()

# ensemble_preds = ensemble_preds / len(saved_model_paths)

# print("\n📊 V7 ENSEMBLE AUC SCORES (14 DISEASES):")
# print("-" * 40)
# individual_aucs = []
# for i, disease in enumerate(CLASSES):
#     try:
#         auc = roc_auc_score(y_true[:, i], ensemble_preds[:, i])
#         individual_aucs.append(auc)
#         print(f"   {disease:20} : {auc:.4f}")
#     except ValueError:
#         print(f"   {disease:20} : N/A")

# macro_auc = np.mean(individual_aucs)
# print("-" * 40)
# print(f"🏆 V7 THESIS ENSEMBLE MACRO AUC: {macro_auc:.4f}")
# print("=" * 50)
# print("🎉 ภารกิจเสร็จสิ้น! V7 พร้อมสำหรับขึ้นพรีเซนต์จบแล้วครับ!")

# ---------------------------------------------------------
# 👑 7️⃣ THE 4TH MODEL: ENSEMBLE EVALUATION & SAVE RESULTS
# ---------------------------------------------------------
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("\n" + "🌟"*25)
print("👑 EVALUATING THE ULTIMATE ENSEMBLE MODEL (V7)")
print("🌟"*25)

y_true = val_generator.labels[:val_generator.samples]
ensemble_preds = np.zeros((val_generator.samples, len(CLASSES)))

for path in saved_model_paths:
    print(f"⏳ Running Inference with: {os.path.basename(path)}")
    model = tf.keras.models.load_model(path, compile=False)
    preds = model.predict(val_generator, steps=len(val_generator), verbose=1)
    ensemble_preds += preds[:val_generator.samples]
    del model
    tf.keras.backend.clear_session()

ensemble_preds = ensemble_preds / len(saved_model_paths)

# ==========================================
# 💾 1. คำนวณและเซฟคะแนน AUC (เป็นไฟล์ .txt)
# ==========================================
print("\n📊 V7 ENSEMBLE AUC SCORES (14 DISEASES):")
print("-" * 40)
individual_aucs = []
auc_log_text = "V7 ENSEMBLE AUC SCORES (14 DISEASES):\n" + "-"*40 + "\n"

for i, disease in enumerate(CLASSES):
    try:
        auc = roc_auc_score(y_true[:, i], ensemble_preds[:, i])
        individual_aucs.append(auc)
        line = f"   {disease:20} : {auc:.4f}"
        print(line)
        auc_log_text += line + "\n"
    except ValueError:
        individual_aucs.append(0.0)
        line = f"   {disease:20} : N/A"
        print(line)
        auc_log_text += line + "\n"

macro_auc = np.mean(individual_aucs)
summary_line = f"\n🏆 V7 THESIS ENSEMBLE MACRO AUC: {macro_auc:.4f}\n"
print("-" * 40)
print(summary_line)
auc_log_text += "-"*40 + summary_line

# เขียนไฟล์ AUC_Scores.txt
with open(os.path.join(GRAPH_SAVE_DIR, 'V7_AUC_Scores.txt'), 'w') as f:
    f.write(auc_log_text)

# ==========================================
# 💾 2. คำนวณและเซฟ Classification Report (เป็นไฟล์ .txt)
# ==========================================
y_pred_binary = (ensemble_preds > 0.5).astype(int)
report = classification_report(y_true, y_pred_binary, target_names=CLASSES, zero_division=0)

# เขียนไฟล์ Classification_Report.txt
with open(os.path.join(GRAPH_SAVE_DIR, 'V7_Classification_Report.txt'), 'w') as f:
    f.write("V7 ENSEMBLE - CLASSIFICATION REPORT (Threshold 0.5)\n\n")
    f.write(report)

# ==========================================
# 💾 3. วาดและเซฟ Confusion Matrix (เป็นรูป .png ครบ 14 โรค)
# ==========================================
cm_dir = os.path.join(GRAPH_SAVE_DIR, 'Confusion_Matrices')
os.makedirs(cm_dir, exist_ok=True)

for i, disease in enumerate(CLASSES):
    cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix: {disease}')
    plt.ylabel('Actual (หมอเฉลย)')
    plt.xlabel('Predicted (AI ทาย)')
    
    # เซฟรูป
    plt.savefig(os.path.join(cm_dir, f'CM_{disease}.png'), bbox_inches='tight', dpi=150)
    plt.close()

print(f"\n✅ เซฟผลการทดลองทั้งหมด (AUC, Report, CM) ไว้ที่โฟลเดอร์: {GRAPH_SAVE_DIR} เรียบร้อยแล้วครับ!")
print("=" * 50)