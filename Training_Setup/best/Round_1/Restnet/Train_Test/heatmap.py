# heatmap.py (FULL FILE) - ResNet Grad-CAM (RUN-PASS) + Top-1 Focus
# ใช้กับโปรเจกต์ของคุณ: models/resnet_best.keras, models/labels.txt, archive/..., splits/test.csv
# รัน: python3 heatmap.py
# ผลลัพธ์: outputs_gradcam/gradcam_001_<top1label>.png

import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================
SPLIT_DIR = "splits"
MODEL_DIR = "models"
OUT_DIR = "outputs_gradcam"
IMG_SIZE = 224

TEST_CSV = os.path.join(SPLIT_DIR, "test.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "resnet_best.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")

PATH_COL = "img_path"
LABEL_COL = "Finding Labels"

# ✅ ใส่รูปเองก่อน (ชัวร์สุด)
MANUAL_IMAGE_PATHS: List[str] = [
    "archive/images_001/images/00000008_000.png",
]

# ✅ โฟกัสแบบ Top-1 เสมอ
AUTO_FOCUS_TOP1 = True
FOCUS_LABEL = None

# ✅ ถ้ารู้ชื่อ conv ใส่ได้ (ไม่รู้ปล่อย None)
BACKBONE_CONV_HINT: Optional[str] = None  # แนะนำให้ None ก่อนให้มัน auto-pick

# =====================
# 🔥 HEATMAP TUNING (แก้ "สีฟุ้ง/กระจาย" ให้เป็นก้อนชัด)
# =====================
HEATMAP_SMOOTH = 0.10          # 0 = ไม่ smooth, 0.05-0.20 = smooth เบาๆ (แนะนำ 0.10)
HEATMAP_GAMMA = 3.0            # ยิ่งมาก ยิ่งเน้นจุดร้อนให้เด่น (แนะนำ 2.0-4.0)
HEATMAP_KEEP_TOP_PERCENT = 15  # เก็บเฉพาะ top กี่ % ของค่าที่แรงสุด (แนะนำ 10-25)
HEATMAP_MIN_AREA = 0           # (ทางเลือก) กันจุดเล็กจิ๋ว - 0 = ปิด (เปิดยาก/ไม่จำเป็น)

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
    return s or "focus"


def preprocess_image(img_path: str) -> tf.Tensor:
    b = tf.io.read_file(img_path)
    img = tf.image.decode_image(b, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img  # (H,W,3) 0..1


# =====================
# PICK LAST 4D LAYER (ROBUST)
# =====================
def _match_layer_by_hint(model: tf.keras.Model, hint: str) -> Optional[tf.keras.layers.Layer]:
    if not hint:
        return None

    try:
        return model.get_layer(hint)
    except Exception:
        pass

    h = hint.lower()
    for lyr in model.layers:
        if h in lyr.name.lower():
            return lyr
    return None


def pick_last_4d_layer_by_forward(model: tf.keras.Model, conv_hint: Optional[str] = None) -> tf.keras.layers.Layer:
    dummy = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)

    if conv_hint:
        lyr = _match_layer_by_hint(model, conv_hint)
        if lyr is not None:
            try:
                probe = tf.keras.Model(inputs=model.inputs, outputs=lyr.output)
                y = probe(dummy, training=False)
                if len(y.shape) == 4:
                    print(f"[INFO] Using conv layer from hint: {lyr.name} | shape={y.shape}")
                    return lyr
                else:
                    print(f"[WARN] Hint layer found but not 4D: {lyr.name} | shape={y.shape}")
            except Exception as e:
                print(f"[WARN] Hint layer forward-test failed: {lyr.name} | {e}")

    candidates = []
    for lyr in reversed(model.layers):
        name = lyr.name.lower()
        if ("conv" in name) or ("block" in name) or ("out" in name):
            candidates.append(lyr)

    if not candidates:
        candidates = list(reversed(model.layers))

    tried = 0
    for lyr in candidates:
        tried += 1
        try:
            if getattr(lyr, "output", None) is None:
                continue
            probe = tf.keras.Model(inputs=model.inputs, outputs=lyr.output)
            y = probe(dummy, training=False)
            if len(y.shape) == 4:
                print(f"[INFO] Auto-picked last 4D layer: {lyr.name} | shape={y.shape} | tried={tried}")
                return lyr
        except Exception:
            continue

    tail = [l.name for l in model.layers[-60:]]
    raise ValueError(
        "❌ Auto-pick หา layer 4D ไม่เจอ (forward test ก็ไม่ผ่าน)\n"
        "ลองดูชื่อ layer ท้าย ๆ:\n- " + "\n- ".join(tail)
    )


def build_grad_model(model: tf.keras.Model, conv_layer: tf.keras.layers.Layer) -> tf.keras.Model:
    return tf.keras.Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])


# =====================
# HEATMAP POSTPROCESS (ทำให้ไม่ฟุ้ง)
# =====================
def _smooth_heatmap_np(hm: np.ndarray, strength: float) -> np.ndarray:
    """
    Smooth เบา ๆ ด้วย avg_pool เพื่อลด noise แต่ไม่ให้ฟุ้งเกินไป
    strength: 0..1 (แนะนำ 0.05-0.20)
    """
    if strength <= 0:
        return hm

    k = 3  # kernel เล็กพอ ไม่ทำให้กระจายเยอะ
    x = hm.astype(np.float32)[None, :, :, None]  # (1,H,W,1)
    x = tf.constant(x)
    x = tf.nn.avg_pool2d(x, ksize=k, strides=1, padding="SAME")
    sm = x.numpy()[0, :, :, 0]

    # blend ระหว่างของเดิมกับ smooth
    out = (1 - strength) * hm + strength * sm
    return out


def refine_heatmap(
    heatmap: np.ndarray,
    gamma: float = 3.0,
    keep_top_percent: int = 15,
    smooth_strength: float = 0.10,
) -> np.ndarray:
    """
    ทำให้ heatmap "เป็นก้อน" มากขึ้น:
    1) smooth เบา ๆ
    2) normalize 0..1
    3) gamma (เพิ่มคอนทราสต์)
    4) percentile threshold (ตัดส่วนอ่อนทิ้ง)
    """
    hm = np.array(heatmap, dtype=np.float32)

    # 1) smooth เบา ๆ (ลด noise)
    hm = _smooth_heatmap_np(hm, smooth_strength)

    # 2) normalize
    hm = np.maximum(hm, 0)
    mx = float(np.max(hm)) if np.max(hm) > 0 else 0.0
    hm = hm / (mx + 1e-8)

    # 3) gamma -> เน้น hotspot
    if gamma and gamma > 1.0:
        hm = np.power(hm, gamma)

    # 4) keep top %
    keep_top_percent = int(np.clip(keep_top_percent, 1, 100))
    thr = np.percentile(hm, 100 - keep_top_percent)
    hm = np.where(hm >= thr, hm, 0.0)

    # normalize อีกรอบให้สวย
    mx2 = float(np.max(hm)) if np.max(hm) > 0 else 0.0
    hm = hm / (mx2 + 1e-8)

    return hm


# =====================
# GRAD-CAM
# =====================
def gradcam_heatmap(grad_model: tf.keras.Model, x: tf.Tensor, class_index: int) -> np.ndarray:
    x = tf.cast(x, tf.float32)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        raise RuntimeError(
            "❌ Gradient is None: preds ไม่เชื่อมกับ conv_out\n"
            "ให้ลองตั้ง BACKBONE_CONV_HINT เป็นชื่อที่เห็นใน log เช่น 'conv4_block6_out' หรือ 'conv5_block1_out'\n"
            "หรือปล่อย None แล้วให้ auto-pick ใหม่"
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))        # (C,)
    conv_out = conv_out[0]                                       # (H,W,C)
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)    # (H,W)

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

    probs = model.predict(x, verbose=0)[0]
    topk = np.argsort(probs)[::-1][:5]

    focus_idx = int(topk[0])  # ✅ Top-1
    focus_label = labels[focus_idx]
    focus_percent = probs[focus_idx] * 100

    # raw heatmap
    heatmap = gradcam_heatmap(grad_model, x, focus_idx)

    # ✅ ทำให้ไม่ฟุ้ง/ไม่กระจาย
    heatmap = refine_heatmap(
        heatmap,
        gamma=HEATMAP_GAMMA,
        keep_top_percent=HEATMAP_KEEP_TOP_PERCENT,
        smooth_strength=HEATMAP_SMOOTH,
    )

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
    ax2.set_title(f"AI Focus: {focus_label} ({focus_percent:.2f}%)", fontsize=16, color="red")

    ax_text.axis("off")
    ax_text.text(
        0.5, 0.5, txt,
        ha="center", va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.7", fc="white", ec="#cccccc")
    )

    return fig, focus_label, float(probs[focus_idx])


# =====================
# MAIN
# =====================
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Missing labels: {LABELS_PATH}")

    print("[INFO] Using model:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    labels = load_labels(LABELS_PATH)

    conv_layer = pick_last_4d_layer_by_forward(model, BACKBONE_CONV_HINT)
    grad_model = build_grad_model(model, conv_layer)
    print("[INFO] Grad model ready ✅")

    if MANUAL_IMAGE_PATHS:
        paths = MANUAL_IMAGE_PATHS
    else:
        if not os.path.exists(TEST_CSV):
            raise FileNotFoundError(f"Missing test.csv: {TEST_CSV}")
        df = pd.read_csv(TEST_CSV)
        if PATH_COL not in df.columns:
            raise ValueError(f"test.csv missing column: {PATH_COL}")
        paths = [str(df.iloc[0][PATH_COL])]

    for idx, img_path in enumerate(paths, start=1):
        if not os.path.exists(img_path):
            print(f"[SKIP] not found: {img_path}")
            continue

        fig, focus_label, focus_prob = render_one(img_path, model, labels, grad_model)

        out_name = f"gradcam_{idx:03d}_{safe_name(focus_label)}.png"
        out_path = os.path.join(OUT_DIR, out_name)
        plt.savefig(out_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

        print(f"[OK] image = {img_path}")
        print(f"     Focus: {focus_label} {focus_prob*100:.2f}%")
        print(f"     Saved: {out_path}")

    print("\nDONE")
    print("ดูไฟล์รูปที่:", OUT_DIR)


if __name__ == "__main__":
    main()
