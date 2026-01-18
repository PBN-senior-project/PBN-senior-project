# heatmap_densenet_5cls.py
# Grad-CAM for DenseNet 5-class (Atelectasis, Cardiomegaly, Edema, Emphysema, Fibrosis)
# Run: python heatmap_densenet_5cls.py
# Output: outputs_gradcam/gradcam_001_<top1label>.png

import os
import re
from typing import List, Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================
IMG_SIZE = 224
OUT_DIR = "outputs_gradcam"

MODEL_PATH = "densenet_5cls_best.keras"   # <- ให้ตรงกับไฟล์โมเดลคุณ
LABELS_PATH = "labels_5cls.txt"           # <- ไฟล์ label 5 คลาส

# เลือกภาพแบบใส่เอง (ชัวร์สุด)
MANUAL_IMAGE_PATHS: List[str] = [
    # "archive/images_001/images/00000008_000.png",
]

# หรือเลือกภาพจาก test_list.txt (ถ้า MANUAL ว่าง)
TEST_LIST = os.path.join("archive", "test_list.txt")

# DenseNet121 last conv layer (Grad-CAM)
LAST_CONV_NAME = "conv5_block16_concat"

os.makedirs(OUT_DIR, exist_ok=True)


# =====================
# UTILS
# =====================
def load_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def safe_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s or "label"

def preprocess_image(img_path: str) -> tf.Tensor:
    b = tf.io.read_file(img_path)
    img = tf.image.decode_image(b, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img  # (H,W,3) 0..1

def pick_images_from_list(list_path: str, n: int = 1) -> List[str]:
    if not os.path.exists(list_path):
        return []
    with open(list_path, "r", encoding="utf-8") as f:
        names = [l.strip() for l in f if l.strip()]
    out = []
    # หา path จริงใน images_001..images_012
    for name in names:
        found = None
        for i in range(1, 13):
            p = f"archive/images_{i:03d}/images/{name}"
            if os.path.exists(p):
                found = p
                break
        if found:
            out.append(found)
        if len(out) >= n:
            break
    return out


# =====================
# GRAD-CAM CORE
# =====================
def build_grad_model(model: tf.keras.Model, conv_layer_name: str) -> tf.keras.Model:
    try:
        conv_layer = model.get_layer(conv_layer_name)
    except Exception as e:
        # ถ้าชื่อไม่ตรง ให้ช่วยพิมพ์ชื่อใกล้เคียง
        candidates = [l.name for l in model.layers if "conv5" in l.name.lower() and "concat" in l.name.lower()]
        raise ValueError(
            f"❌ หา layer '{conv_layer_name}' ไม่เจอ: {e}\n"
            f"ลองใช้ชื่อพวกนี้แทน:\n- " + "\n- ".join(candidates[:30])
        )

    return tf.keras.Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])

def gradcam_heatmap(grad_model: tf.keras.Model, x: tf.Tensor, class_index: int) -> np.ndarray:
    x = tf.cast(x, tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)  # conv_out: (B,H,W,C), preds: (B,num_classes)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        raise RuntimeError("❌ Gradient is None (กราฟไม่เชื่อม)")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)
    conv_out = conv_out[0]                                 # (H,W,C)
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)  # (H,W)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(img_np: np.ndarray, heatmap: np.ndarray, alpha=0.45) -> np.ndarray:
    hm = tf.image.resize(heatmap[..., None], (IMG_SIZE, IMG_SIZE)).numpy().squeeze()
    hm = np.clip(hm, 0, 1)
    cmap = plt.get_cmap("jet")
    hm_rgb = cmap(hm)[..., :3]
    overlay = np.clip((1 - alpha) * img_np + alpha * hm_rgb, 0, 1)
    return overlay


# =====================
# RENDER ONE
# =====================
def render_one(img_path: str, model, labels, grad_model):
    img = preprocess_image(img_path)
    img_np = img.numpy()
    x = tf.expand_dims(img, 0)

    probs = model.predict(x, verbose=0)[0]  # (num_classes,)
    topk = np.argsort(probs)[::-1][:5]

    top1 = int(topk[0])
    top1_label = labels[top1]
    top1_prob = float(probs[top1])

    heatmap = gradcam_heatmap(grad_model, x, top1)
    overlay = overlay_heatmap(img_np, heatmap, alpha=0.45)

    lines = ["Analysis Report:"]
    lines.append("------------------------------")
    for i in topk:
        i = int(i)
        lines.append(f"{labels[i]}: {probs[i]*100:.2f}%")
    txt = "\n".join(lines)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[10, 3.2], hspace=0.15, wspace=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])

    ax1.imshow(img_np)
    ax1.axis("off")
    ax1.set_title("Original X-ray", fontsize=16)

    ax2.imshow(overlay)
    ax2.axis("off")
    ax2.set_title(f"AI Focus: {top1_label} ({top1_prob*100:.2f}%)", fontsize=16, color="red")

    ax_text.axis("off")
    ax_text.text(
        0.5, 0.5, txt,
        ha="center", va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.7", fc="white", ec="#cccccc")
    )

    return fig, top1_label, top1_prob


# =====================
# MAIN
# =====================
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Missing labels: {LABELS_PATH}")

    labels = load_labels(LABELS_PATH)

    print("[INFO] Loading model:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("[INFO] Building grad_model with conv layer:", LAST_CONV_NAME)
    grad_model = build_grad_model(model, LAST_CONV_NAME)
    print("[INFO] Grad model ready ✅")

    if MANUAL_IMAGE_PATHS:
        paths = MANUAL_IMAGE_PATHS
    else:
        paths = pick_images_from_list(TEST_LIST, n=1)

    if not paths:
        raise ValueError("❌ ไม่พบรูปให้ทำ heatmap (ใส่ MANUAL_IMAGE_PATHS หรือเช็ค test_list.txt)")

    for idx, img_path in enumerate(paths, start=1):
        if not os.path.exists(img_path):
            print("[SKIP] not found:", img_path)
            continue

        fig, label, prob = render_one(img_path, model, labels, grad_model)

        out_name = f"gradcam_{idx:03d}_{safe_name(label)}.png"
        out_path = os.path.join(OUT_DIR, out_name)
        plt.savefig(out_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

        print(f"[OK] image = {img_path}")
        print(f"     Top-1: {label} {prob*100:.2f}%")
        print(f"     Saved: {out_path}")

    print("\nDONE")
    print("ดูไฟล์รูปที่:", OUT_DIR)


if __name__ == "__main__":
    main()
