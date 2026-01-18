# test_evaluate.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.resnet50 import preprocess_input

# =====================
# CONFIG
# =====================
SPLIT_DIR = "splits"
MODEL_DIR = "models"
OUT_DIR = "outputs_eval"

TEST_CSV = os.path.join(SPLIT_DIR, "test.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "resnet_best.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")

PATH_COL = "img_path"
LABEL_COL = "Finding Labels"

IMG_SIZE = 224
BATCH_SIZE = 32

os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# Helpers
# =====================
def load_labels(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing labels file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if len(labels) < 2:
        raise RuntimeError(f"labels.txt must contain >=2 labels, got {len(labels)}")
    return labels

def first_label(label_str: str) -> str:
    """ถ้า label เป็น 'A|B|C' จะเอา 'A'"""
    if label_str is None:
        return ""
    s = str(label_str).strip()
    if not s:
        return ""
    return s.split("|")[0].strip()

def decode_and_preprocess(path: tf.Tensor, label: tf.Tensor):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)  # 0..255
    img = preprocess_input(img)     # ResNet50 preprocess
    return img, label

def make_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def plot_confusion_matrix(cm: np.ndarray, class_names, save_path: str, title: str):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # ใส่ตัวเลขในช่อง
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9
            )

    plt.ylabel("True label (ความจริง)")
    plt.xlabel("Predicted label (AI ทาย)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()

# =====================
# Main
# =====================
def main():
    # --- check files ---
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Missing: {TEST_CSV}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing: {MODEL_PATH}")

    labels = load_labels(LABELS_PATH)
    label2idx = {name: i for i, name in enumerate(labels)}

    # --- load test csv ---
    df = pd.read_csv(TEST_CSV)
    if PATH_COL not in df.columns:
        raise ValueError(f"test.csv missing column: {PATH_COL}")
    if LABEL_COL not in df.columns:
        raise ValueError(f"test.csv missing column: {LABEL_COL}")

    df = df.copy()
    df[PATH_COL] = df[PATH_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(str).apply(first_label)

    # เอาเฉพาะแถวที่มีไฟล์จริง
    df = df[df[PATH_COL].apply(os.path.exists)].copy()
    # เอาเฉพาะ label ที่อยู่ใน labels.txt เท่านั้น
    df = df[df[LABEL_COL].isin(labels)].copy()

    if len(df) == 0:
        raise RuntimeError(
            "No valid test rows after filtering.\n"
            "- check img_path exists\n"
            "- check labels in labels.txt matches Finding Labels"
        )

    df["y_true"] = df[LABEL_COL].map(label2idx).astype(int)

    # --- dataset ---
    test_paths = df[PATH_COL].values
    test_y = df["y_true"].values
    test_ds = make_dataset(test_paths, test_y)

    # --- load model ---
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # --- predict ---
    probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    # --- report ---
    print("\n==================== Classification Report ====================")
    print(classification_report(
        test_y, y_pred,
        target_names=labels,
        digits=2
    ))

    # --- confusion matrix ---
    cm = confusion_matrix(test_y, y_pred, labels=list(range(len(labels))))

    print("\n==================== Confusion Matrix (แนวตั้ง=ความจริง, แนวนอน=AI ทาย) ====================")
    header = " " * 14 + " ".join([f"{name:>12}" for name in labels])
    print(header)
    for i, name in enumerate(labels):
        row_vals = " ".join([f"{cm[i, j]:>12d}" for j in range(len(labels))])
        print(f"{name:>12}  {row_vals}")

    # save confusion matrix image
    out_png = os.path.join(OUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(
        cm,
        class_names=labels,
        save_path=out_png,
        title="Confusion Matrix (True vs Pred)"
    )

    print("\n✅ Saved confusion matrix image:", out_png)

    # (optional) save predictions csv
    out_pred_csv = os.path.join(OUT_DIR, "test_predictions.csv")
    df_out = df[[PATH_COL, LABEL_COL]].copy()
    df_out["y_true"] = test_y
    df_out["pred_label"] = [labels[i] for i in y_pred]
    df_out["pred_prob"] = probs[np.arange(len(probs)), y_pred]
    df_out.to_csv(out_pred_csv, index=False)
    print("✅ Saved predictions:", out_pred_csv)

if __name__ == "__main__":
    main()
