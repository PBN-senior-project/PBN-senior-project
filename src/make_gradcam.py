import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def list_images_recursive(root_dir: Path, exts=(".png", ".jpg", ".jpeg")) -> Dict[str, str]:
    mapping = {}
    for fp in root_dir.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in exts:
            mapping[fp.name] = str(fp.resolve())
    return mapping

def load_list_file(list_path: str) -> List[str]:
    lines = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(Path(line).name)
    return lines

def labels_to_multihot(finding_labels: str, classes: List[str]) -> np.ndarray:
    s = str(finding_labels) if finding_labels is not None else ""
    parts = [x.strip() for x in s.split("|") if x.strip()]
    parts_set = set(parts)
    y = np.zeros((len(classes),), dtype=np.float32)
    for i, c in enumerate(classes):
        if c in parts_set:
            y[i] = 1.0
    return y

def load_and_preprocess_image(path: str, img_size: int) -> Tuple[np.ndarray, np.ndarray]:
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img_resized = tf.image.resize(img, (img_size, img_size), antialias=True)
    return img.numpy(), img_resized.numpy()

def build_val_df(cfg: dict) -> Tuple[pd.DataFrame, List[str]]:
    classes = cfg["classes"]
    df = pd.read_csv(cfg["csv_path"])
    df.columns = [c.strip() for c in df.columns]

    img_root = Path(cfg["img_root"])
    img_map = list_images_recursive(img_root)

    df["Image Index"] = df["Image Index"].astype(str).str.strip()
    df["path"] = df["Image Index"].map(img_map.get)

    miss = df["path"].isna()
    if miss.any():
        alt = df.loc[miss, "Image Index"].str.replace(".png", ".jpg", regex=False)
        df.loc[miss, "path"] = alt.map(img_map.get)

    miss = df["path"].isna()
    if miss.any():
        alt = df.loc[miss, "Image Index"].str.replace(".jpg", ".png", regex=False)
        df.loc[miss, "path"] = alt.map(img_map.get)

    df = df[df["path"].notna()].reset_index(drop=True)

    train_list = cfg.get("train_list", None)
    if train_list and Path(train_list).exists():
        train_names = set(load_list_file(train_list))
        df_train = df[df["Image Index"].isin(train_names)].copy()
    else:
        df_train = df.copy()

    df_train = df_train.sort_values("Image Index").reset_index(drop=True)

    train_df, val_df = train_test_split(
        df_train,
        test_size=float(cfg.get("val_size", 0.2)),
        random_state=int(cfg.get("seed", 42)),
        shuffle=True,
    )

    return val_df.reset_index(drop=True), classes

def find_last_conv_layer_name(model: keras.Model) -> str:
    # หา Conv2D ตัวสุดท้ายแบบอัตโนมัติ (ใช้ได้กับ DenseNet/ResNet/MobileNet ส่วนใหญ่)
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    # กรณีเป็น nested model (applications) อาจซ่อนอยู่ข้างใน
    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, keras.layers.Conv2D):
                    return sub.name
    raise ValueError("Cannot find a Conv2D layer for Grad-CAM.")

def gradcam_heatmap(model: keras.Model, img_batch: tf.Tensor, class_index: int, conv_layer_name: str) -> np.ndarray:
    conv_layer = model.get_layer(conv_layer_name)
    grad_model = keras.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch, training=False)
        # multi-label: ใช้คะแนนของคลาสนั้นโดยตรง
        class_score = preds[:, class_index]

    grads = tape.gradient(class_score, conv_out)  # (1,H,W,C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_out = conv_out[0]  # (H,W,C)
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)  # (H,W)

    heatmap = tf.maximum(heatmap, 0)  # ReLU
    denom = tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap / denom
    return heatmap.numpy()

def overlay_and_save(orig_img01: np.ndarray, heatmap: np.ndarray, out_path: Path, title: str):
    # orig_img01: [H,W,3] in [0,1]
    plt.figure()
    plt.imshow(orig_img01)
    plt.imshow(heatmap, alpha=0.35)  # overlay
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main(run_name="baseline_run", use_best=True, n_images=10, thr=0.5):
    run_dir = project_root() / "runs" / run_name
    cfg = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    classes = cfg["classes"]
    img_size = int(cfg["img_size"])

    model_path = run_dir / ("best.keras" if use_best else "final.keras")
    model = keras.models.load_model(str(model_path), compile=False)

    conv_name = find_last_conv_layer_name(model)
    print("Using last conv layer for Grad-CAM:", conv_name)

    val_df, _ = build_val_df(cfg)

    # สุ่มภาพ
    val_df = val_df.sample(n=min(n_images, len(val_df)), random_state=cfg.get("seed", 42)).reset_index(drop=True)

    out_dir = run_dir / "gradcam_outputs"
    ensure_dir(out_dir)

    for i in range(len(val_df)):
        img_path = val_df.loc[i, "path"]
        img_index = val_df.loc[i, "Image Index"]
        finding = str(val_df.loc[i, "Finding Labels"])

        orig01, resized01 = load_and_preprocess_image(img_path, img_size)  # orig01 may be different size
        inp = tf.convert_to_tensor(resized01[None, ...], dtype=tf.float32)  # (1,img,img,3)

        preds = model.predict(inp, verbose=0)[0]  # (C,)
        top_idx = int(np.argmax(preds))
        top_score = float(preds[top_idx])

        # ถ้าคะแนนต่ำมาก อาจไม่ชัด: คุณปรับ thr ได้
        if top_score < thr:
            # ก็ยังทำได้อยู่ แค่แจ้งไว้ในชื่อไฟล์
            pass

        heat = gradcam_heatmap(model, inp, top_idx, conv_name)
        # resize heatmap ให้เท่ากับภาพต้นฉบับ
        heat_tf = tf.image.resize(heat[..., None], (orig01.shape[0], orig01.shape[1]), antialias=True)
        heat_resized = heat_tf.numpy().squeeze()

        title = f"{img_index} | top={classes[top_idx]}:{top_score:.3f} | labels={finding}"
        out_path = out_dir / f"gradcam_{i:03d}_{classes[top_idx]}_{img_index}.png"
        overlay_and_save(orig01, heat_resized, out_path, title)
        print("[OK]", out_path)

    print("Saved Grad-CAM overlays to:", out_dir)


if __name__ == "__main__":
    # ✅ เปลี่ยน run_name ให้ตรงกับของคุณ
    main(run_name="baseline_run", use_best=True, n_images=10, thr=0.5)
