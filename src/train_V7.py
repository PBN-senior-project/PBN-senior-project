import os, glob, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os, glob, cv2
import time
import numpy as np

from tensorflow.keras import layers, models, optimizers, callbacks, mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, ResNet50V2, MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


# =========================================================
# CONFIG
# =========================================================
# mixed_precision.set_global_policy("mixed_float16") for GPU
mixed_precision.set_global_policy("float32")

CSV_PATH = "/archive/Data_Entry_2017.csv"
ARCHIVE_DIR = "/archive"
MODEL_SAVE_DIR = "/app/models_v7"
GRAPH_SAVE_DIR = "/app/outputs/graphs_v7"

IMG_SIZE = (384, 384)
BATCH_SIZE = 8
EPOCHS = 10
INITIAL_LR = 1e-4
N_NO_FINDING = 5000
SEED = 42

CLASSES = [
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Nodule",
    "Mass",
    "Pneumothorax"
]

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(GRAPH_SAVE_DIR, exist_ok=True)

# for gpu in tf.config.list_physical_devices("GPU"):
#     tf.config.experimental.set_memory_growth(gpu, True)


# =========================================================
# 1. LOAD + SELECT DATA
# =========================================================
print("⏳ Loading Data...")
df = pd.read_csv(CSV_PATH)

image_paths = glob.glob(
    os.path.join(ARCHIVE_DIR, "**", "*.png"),
    recursive=True
)

image_map = {os.path.basename(p): p for p in image_paths}
df["filepath"] = df["Image Index"].map(image_map)

print(f"📁 Images found : {len(image_paths)}")
print(f"✅ Matched      : {df['filepath'].notna().sum()}")
print(f"❌ Missing      : {df['filepath'].isna().sum()}")

# เอาเฉพาะภาพที่มีไฟล์จริง
df = df[df["filepath"].notna()].copy()

# ใช้ View Position จาก CSV โดยตรง และเอาเฉพาะ PA / AP
df["View Position"] = df["View Position"].astype(str).str.strip()
df = df[df["View Position"].isin(["PA", "AP"])].copy()

print("\n📌 View Position:")
print(df["View Position"].value_counts())

# สร้าง Multi-label สำหรับ 6 findings
labels = df["Finding Labels"].fillna("").str.split("|")

for cls in CLASSES:
    df[cls] = labels.apply(lambda x: int(cls in x))

# Positive = มีอย่างน้อย 1 ใน 6 findings
positive_df = df[df[CLASSES].sum(axis=1) > 0].copy()

# Negative = No Finding
negative_df = df[df["Finding Labels"].eq("No Finding")].copy()

# จำกัด No Finding
negative_df = negative_df.sample(
    n=min(N_NO_FINDING, len(negative_df)),
    random_state=SEED
)

# รวม Positive + Negative แล้ว shuffle
selected_df = pd.concat(
    [positive_df, negative_df],
    ignore_index=True
).sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

print(f"\n📦 Selected : {len(selected_df)} images")
print(f"👤 Patients : {selected_df['Patient ID'].nunique()}")


# =========================================================
# 2. PATIENT-LEVEL SPLIT
# =========================================================
patients = selected_df["Patient ID"].unique()

train_patients, val_patients = train_test_split(
    patients,
    test_size=0.2,
    random_state=SEED
)

train_df = selected_df[
    selected_df["Patient ID"].isin(train_patients)
].copy()

val_df = selected_df[
    selected_df["Patient ID"].isin(val_patients)
].copy()

overlap = set(train_df["Patient ID"]) & set(val_df["Patient ID"])
assert not overlap, "❌ Patient leakage detected!"

print(f"\n✅ Train : {len(train_df)}")
print(f"✅ Val   : {len(val_df)}")
print(f"✅ Patient overlap : {len(overlap)}")


def show_distribution(data, name):
    print(f"\n📊 {name}")
    for cls in CLASSES:
        pos = data[data[cls] == 1]["View Position"].value_counts()
        pa, ap = pos.get("PA", 0), pos.get("AP", 0)
        print(f"{cls:15} PA={pa:5} AP={ap:5} Total={pa + ap:5}")


show_distribution(train_df, "TRAIN BEFORE OVERSAMPLING")
show_distribution(val_df, "VALIDATION")


# =========================================================
# 3. TARGETED OVERSAMPLING — TRAIN ONLY
# =========================================================
before = len(train_df)

class_counts = train_df[CLASSES].sum()
minority_classes = class_counts[class_counts < 2000].index.tolist()

print(f"\n🎯 Minority classes: {minority_classes}")

if minority_classes:
    minority_df = train_df[
        train_df[minority_classes].sum(axis=1) > 0
    ]

    train_df = pd.concat(
        [train_df, minority_df],
        ignore_index=True
    ).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"📈 Train: {before} → {len(train_df)} images")


# =========================================================
# 4. WEIGHTED FOCAL LOSS
# =========================================================
pos_counts = train_df[CLASSES].sum().values
weights = len(train_df) / (2.0 * (pos_counts + 1e-5))
pos_weights = tf.constant(weights, dtype=tf.float32)


def weighted_focal_loss(pos_weights, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(
            tf.cast(y_pred, tf.float32),
            1e-7,
            1 - 1e-7
        )

        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha = y_true * pos_weights + (1 - y_true)

        return tf.reduce_mean(
            -alpha * tf.pow(1 - p_t, gamma) * tf.math.log(p_t)
        )
    return loss


# =========================================================
# 5. IMAGE PREPROCESSING
# =========================================================
def apply_clahe(img):
    img = np.clip(img, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(l)

    return cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2RGB
    )


def medical_preprocessing(img):
    img = apply_clahe(img)

    # Random crop 384 → 368 → 384
    h, w = img.shape[:2]
    ch, cw = min(368, h), min(368, w)

    top = np.random.randint(0, h - ch + 1) if h > ch else 0
    left = np.random.randint(0, w - cw + 1) if w > cw else 0

    img = img[top:top + ch, left:left + cw]
    img = cv2.resize(img, IMG_SIZE).astype(np.float32) / 255.0

    # Gamma
    img = np.power(img, np.random.uniform(0.85, 1.15))

    # Gaussian Noise 50%
    if np.random.rand() < 0.5:
        img += np.random.normal(
            0, 0.01, img.shape
        ).astype(np.float32)

    return np.clip(img, 0, 1)


def val_preprocessing(img):
    return apply_clahe(img).astype(np.float32) / 255.0


# =========================================================
# 6. DATA GENERATORS
# =========================================================
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    shear_range=5,
    horizontal_flip=True,
    fill_mode="constant",
    cval=0,
    preprocessing_function=medical_preprocessing
)

val_datagen = ImageDataGenerator(
    preprocessing_function=val_preprocessing
)

generator_args = dict(
    directory=None,
    x_col="filepath",
    y_col=CLASSES,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw"
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    shuffle=True,
    **generator_args
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    shuffle=False,
    **generator_args
)


# =========================================================
# 7. MODEL
# =========================================================
BACKBONES = {
    "DenseNet121": DenseNet121,
    "ResNet50V2": ResNet50V2,
    "MobileNetV2": MobileNetV2
}


def build_model(name):
    base = BACKBONES[name](
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3)
    )

    base.trainable = True

    for layer in base.layers[:-120]:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(
        len(CLASSES),
        activation="sigmoid",
        dtype="float32"
    )(x)

    model = models.Model(base.input, output, name=name)

    model.compile(
        optimizer=optimizers.AdamW(
            learning_rate=INITIAL_LR,
            weight_decay=1e-4
        ),
        loss=weighted_focal_loss(pos_weights),
        metrics=[
            tf.keras.metrics.AUC(
                multi_label=True,
                name="auc"
            )
        ]
    )

    return model

# =========================================================
# GRAFANA / PROMETHEUS TRAINING MONITOR
# =========================================================
class PrometheusTrainingCallback(tf.keras.callbacks.Callback):

    def __init__(self, model_name, total_images, batch_size,
                 total_epochs, total_batches, val_batches):
        super().__init__()

        self.name = model_name
        self.total_images = total_images
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.val_batches = val_batches

        self.model_start = None
        self.epoch_start = None
        self.val_start = None
        self.current_epoch = 0
        self.val_times = []

        self.registry = CollectorRegistry()

        # Prometheus metrics
        metric_names = [
            "training_status",
            "training_epoch",
            "training_total_epochs",
            "training_batch",
            "training_total_batches",
            "training_images_processed",
            "training_total_images",
            "training_progress_percent",
            "training_loss",
            "training_auc",
            "validation_loss",
            "validation_auc",
            "training_seconds_per_step",
            "training_epoch_elapsed_seconds",
            "training_train_remaining_seconds",
            "training_validation_estimate_seconds",
            "training_next_epoch_eta_timestamp",
            "training_model_elapsed_seconds",
            "training_model_remaining_seconds",
            "training_model_finish_timestamp",
            "training_last_update_timestamp"
        ]

        self.m = {
            name: Gauge(name, name, ["model"], registry=self.registry)
            for name in metric_names
        }

    def set(self, name, value):
        self.m[name].labels(model=self.name).set(float(value))

    def push(self):
        try:
            push_to_gateway(
                "pushgateway:9091",
                job="chestxray_training",
                registry=self.registry
            )
        except Exception as e:
            print("⚠️ Prometheus:", e)

    # ---------------- MODEL START ----------------
    def on_train_begin(self, logs=None):
        self.model_start = time.time()

        for name, value in {
            "training_status": 1,
            "training_total_epochs": self.total_epochs,
            "training_total_batches": self.total_batches,
            "training_total_images": self.total_images,
            "training_last_update_timestamp": self.model_start
        }.items():
            self.set(name, value)

        self.push()

    # ---------------- EPOCH START ----------------
    def on_epoch_begin(self, epoch, logs=None):
        now = time.time()
        self.current_epoch = epoch + 1
        self.epoch_start = now

        for name, value in {
            "training_status": 1,
            "training_epoch": self.current_epoch,
            "training_batch": 0,
            "training_images_processed": 0,
            "training_progress_percent": 0,
            "training_last_update_timestamp": now
        }.items():
            self.set(name, value)

        self.push()

    # ---------------- TRAIN BATCH ----------------
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        now = time.time()
        b = batch + 1

        elapsed = now - self.epoch_start
        sec_step = elapsed / b

        remaining = max(self.total_batches - b, 0) * sec_step
        images = min(b * self.batch_size, self.total_images)
        progress = b / self.total_batches * 100

        # Epoch 1 ใช้การประมาณ / Epoch ต่อไปใช้เวลา validation จริง
        val_est = (
            sum(self.val_times) / len(self.val_times)
            if self.val_times
            else self.val_batches * sec_step
        )

        next_epoch = now + remaining + val_est

        epoch_est = self.total_batches * sec_step + val_est
        future_epochs = max(self.total_epochs - self.current_epoch, 0)

        model_remaining = (
            remaining +
            val_est +
            future_epochs * epoch_est
        )

        values = {
            "training_batch": b,
            "training_images_processed": images,
            "training_progress_percent": progress,
            "training_seconds_per_step": sec_step,
            "training_epoch_elapsed_seconds": elapsed,
            "training_train_remaining_seconds": remaining,
            "training_validation_estimate_seconds": val_est,
            "training_next_epoch_eta_timestamp": next_epoch,
            "training_model_elapsed_seconds": now - self.model_start,
            "training_model_remaining_seconds": model_remaining,
            "training_model_finish_timestamp": now + model_remaining,
            "training_last_update_timestamp": now
        }

        if "loss" in logs:
            values["training_loss"] = logs["loss"]

        if "auc" in logs:
            values["training_auc"] = logs["auc"]

        for name, value in values.items():
            self.set(name, value)

        # ส่งทุก 50 batch
        if b == 1 or b % 50 == 0 or b == self.total_batches:
            self.push()

    # ---------------- VALIDATION ----------------
    def on_test_begin(self, logs=None):
        self.val_start = time.time()
        self.set("training_status", 2)
        self.set("training_last_update_timestamp", self.val_start)
        self.push()

    def on_test_end(self, logs=None):
        now = time.time()

        if self.val_start:
            self.val_times.append(now - self.val_start)
            self.set(
                "training_validation_estimate_seconds",
                sum(self.val_times) / len(self.val_times)
            )

        self.set("training_status", 1)
        self.set("training_last_update_timestamp", now)
        self.push()

    # ---------------- EPOCH END ----------------
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        metric_map = {
            "loss": "training_loss",
            "auc": "training_auc",
            "val_loss": "validation_loss",
            "val_auc": "validation_auc"
        }

        for keras_name, metric_name in metric_map.items():
            if keras_name in logs:
                self.set(metric_name, logs[keras_name])

        self.set("training_last_update_timestamp", time.time())
        self.push()

    # ---------------- MODEL END ----------------
    def on_train_end(self, logs=None):
        now = time.time()

        self.set("training_status", 3)
        self.set("training_model_remaining_seconds", 0)
        self.set("training_model_finish_timestamp", now)
        self.set("training_last_update_timestamp", now)

        self.push()

# =========================================================
# 8. TRAIN 3 MODELS
# =========================================================
MODEL_NAMES = ["DenseNet121", "ResNet50V2", "MobileNetV2"]
saved_paths = []
history_dict = {}

for name in MODEL_NAMES:
    print(f"\n🧠 Training {name}")

    model = build_model(name)

    monitor_cb = PrometheusTrainingCallback(
        model_name=name,
        total_images=len(train_df),
        batch_size=BATCH_SIZE,
        total_epochs=EPOCHS,
        total_batches=len(train_generator),
        val_batches=len(val_generator)
    )

    path = os.path.join(
        MODEL_SAVE_DIR,
        f"best_{name}_v7.keras"
    )

    saved_paths.append(path)

    cb = [
        callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True
        ),

        callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.1,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),

        callbacks.ModelCheckpoint(
            path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        monitor_cb
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=cb
    )

    history_dict[name] = history.history["val_auc"]

    del model
    tf.keras.backend.clear_session()


# =========================================================
# 9. TRAINING GRAPH
# =========================================================
plt.figure(figsize=(10, 6))

for name, values in history_dict.items():
    plt.plot(values, marker="o", label=name)

plt.title("V7 Validation AUC Comparison (6 Findings)")
plt.xlabel("Epoch")
plt.ylabel("Validation AUC")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(
    os.path.join(
        GRAPH_SAVE_DIR,
        "training_comparison_v7.png"
    ),
    bbox_inches="tight",
    dpi=300
)

plt.close()


# =========================================================
# 10. ENSEMBLE
# =========================================================
y_true = val_generator.labels[:val_generator.samples]
ensemble_preds = np.zeros(
    (val_generator.samples, len(CLASSES))
)

for path in saved_paths:
    print(f"⏳ Predicting: {os.path.basename(path)}")

    model = tf.keras.models.load_model(
        path,
        compile=False
    )

    preds = model.predict(
        val_generator,
        verbose=1
    )

    ensemble_preds += preds[:val_generator.samples]

    del model
    tf.keras.backend.clear_session()

ensemble_preds /= len(saved_paths)


# =========================================================
# 11. AUC
# =========================================================
auc_lines = ["V7 ENSEMBLE AUC SCORES (6 FINDINGS)", "-" * 40]
aucs = []

for i, disease in enumerate(CLASSES):
    try:
        auc = roc_auc_score(
            y_true[:, i],
            ensemble_preds[:, i]
        )
        aucs.append(auc)
        auc_lines.append(f"{disease:20}: {auc:.4f}")

    except ValueError:
        auc_lines.append(f"{disease:20}: N/A")

macro_auc = np.mean(aucs) if aucs else float("nan")
auc_lines.append(f"\nMACRO AUC: {macro_auc:.4f}")

auc_text = "\n".join(auc_lines)
print("\n" + auc_text)

with open(
    os.path.join(GRAPH_SAVE_DIR, "V7_AUC_Scores.txt"),
    "w"
) as f:
    f.write(auc_text)


# =========================================================
# 12. CLASSIFICATION REPORT + CONFUSION MATRICES
# =========================================================
y_pred = (ensemble_preds > 0.5).astype(int)

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASSES,
    zero_division=0
)

with open(
    os.path.join(
        GRAPH_SAVE_DIR,
        "V7_Classification_Report.txt"
    ),
    "w"
) as f:
    f.write(
        "V7 ENSEMBLE - CLASSIFICATION REPORT "
        "(Threshold 0.5)\n\n"
    )
    f.write(report)


cm_dir = os.path.join(
    GRAPH_SAVE_DIR,
    "Confusion_Matrices"
)
os.makedirs(cm_dir, exist_ok=True)

for i, disease in enumerate(CLASSES):
    cm = confusion_matrix(
        y_true[:, i],
        y_pred[:, i]
    )

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )

    plt.title(f"Confusion Matrix: {disease}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    plt.savefig(
        os.path.join(cm_dir, f"CM_{disease}.png"),
        bbox_inches="tight",
        dpi=150
    )
    plt.close()

print(f"\n✅ Results saved: {GRAPH_SAVE_DIR}")