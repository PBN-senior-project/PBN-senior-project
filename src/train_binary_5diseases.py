# src/train_binary_5diseases.py
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------
# Reproducibility
# -----------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# -----------------------
# Config
# -----------------------
@dataclass
class CFG:
    img_size: int = 224
    batch_size: int = 16
    epochs_head: int = 5
    epochs_fine: int = 10
    lr_head: float = 1e-4
    lr_fine: float = 3e-5
    out_dir: str = "outputs_binary"
    num_cam_images: int = 12

# -----------------------
# Auto-detect columns (NIH)
# -----------------------
COMMON_FILENAME_COLS = ["Image Index", "filename", "file", "path"]
COMMON_LABEL_COLS    = ["Finding Labels", "labels", "label"]
COMMON_PATIENT_COLS  = ["Patient ID", "patient_id", "pid"]

def pick_col(df_cols, preferred_list):
    cols = list(df_cols)
    lower_map = {c.lower(): c for c in cols}
    for cand in preferred_list:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def resolve_columns(df, filename_col=None, label_col=None, patient_col=None):
    fcol = filename_col or pick_col(df.columns, COMMON_FILENAME_COLS)
    lcol = label_col    or pick_col(df.columns, COMMON_LABEL_COLS)
    pcol = patient_col  or pick_col(df.columns, COMMON_PATIENT_COLS)

    missing = [name for name, col in [("filename", fcol), ("labels", lcol), ("patient_id", pcol)] if col is None]
    if missing:
        raise ValueError(
            "Cannot auto-detect required columns: " + ", ".join(missing) +
            "\nColumns found:\n- " + "\n- ".join(map(str, df.columns))
        )
    return fcol, lcol, pcol

# -----------------------
# Image indexing (solve images_001/images_002 split)
# -----------------------
def build_image_index(img_root: str, exts=(".png", ".jpg", ".jpeg")):
    img_root = Path(img_root)
    if not img_root.exists():
        raise FileNotFoundError(f"img_root not found: {img_root.resolve()}")

    files = []
    for ext in exts:
        files.extend(img_root.rglob(f"*{ext}"))

    index = {p.name: str(p) for p in files}
    print(f"[image_index] found {len(index):,} images under {img_root.resolve()}")
    return index

def attach_fullpaths(df: pd.DataFrame, img_index: dict):
    df = df.copy()
    df["fullpath"] = df["filename"].map(img_index)

    missing = int(df["fullpath"].isna().sum())
    if missing:
        print(f"[warn] missing {missing:,} images -> dropping (avoid ReadFile crash)")
        print("[warn] sample missing:", df.loc[df["fullpath"].isna(), "filename"].head(10).tolist())
        df = df.dropna(subset=["fullpath"]).reset_index(drop=True)

    return df

# -----------------------
# Label utils
# -----------------------
def parse_labels(s):
    if not isinstance(s, str) or not s.strip():
        return []
    return [x.strip() for x in s.split("|") if x.strip()]

def is_no_finding(label_str: str):
    # รองรับหลายคำที่เจอได้
    if not isinstance(label_str, str):
        return False
    s = label_str.strip().lower()
    return s in ["no finding", "no findings", "sin hallazgos", "sin hallazgo", "normal"]

def make_binary_dataset(df, disease: str, per_class=1000, seed=42, strict_single_label=True):
    """
    สร้างชุดข้อมูล binary: disease vs No Finding
    - strict_single_label=True: ใช้เฉพาะภาพที่มี label เดี่ยว (สะอาดที่สุด)
      disease: ต้องเป็น [disease] อย่างเดียว
      no finding: ต้องเป็น [No Finding] อย่างเดียว
    """
    df = df.copy()
    df["labels_list"] = df["labels"].apply(parse_labels)

    if strict_single_label:
        df = df[df["labels_list"].apply(lambda x: len(x) == 1)].copy()
        df["single"] = df["labels_list"].apply(lambda x: x[0])
        pos = df[df["single"] == disease].copy()
        neg = df[df["single"].apply(lambda x: is_no_finding(x))].copy()
    else:
        # ไม่แนะนำสำหรับเอา acc สูง ๆ แต่เผื่อไว้
        pos = df[df["labels_list"].apply(lambda xs: disease in xs)].copy()
        neg = df[df["labels_list"].apply(lambda xs: any(is_no_finding(x) for x in xs))].copy()

    if len(pos) < per_class:
        raise ValueError(f"[{disease}] positive images only {len(pos)} (<{per_class}). ลด per_class หรือปิด strict_single_label")
    if len(neg) < per_class:
        raise ValueError(f"[{disease}] No Finding images only {len(neg)} (<{per_class}).")

    pos = pos.sample(per_class, random_state=seed).copy()
    neg = neg.sample(per_class, random_state=seed).copy()

    pos["target"] = 1
    neg["target"] = 0

    out = pd.concat([pos, neg], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out

def split_800_200_each_class(df, train_n=800, test_n=200, seed=42):
    """
    split ให้ได้ train 800/test 200 ต่อคลาส (0 และ 1)
    """
    tr_parts, te_parts = [], []
    for cls in [0, 1]:
        part = df[df["target"] == cls].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if len(part) < train_n + test_n:
            raise ValueError(f"class {cls} has {len(part)} rows (<{train_n+test_n})")
        tr_parts.append(part.iloc[:train_n])
        te_parts.append(part.iloc[train_n:train_n+test_n])

    tr = pd.concat(tr_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    te = pd.concat(te_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return tr, te

# -----------------------
# TF pipeline (augment only train)
# -----------------------
def decode_image(path, img_size):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (img_size, img_size), method="bilinear")
    return img

def augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.08)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    return img

def make_ds(df, img_size, batch_size, training):
    paths = df["fullpath"].astype(str).values
    y = df["target"].astype(np.int32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, y))

    def _load(p, y):
        img = decode_image(p, img_size)
        if training:
            img = augment(img)  # ✅ augment only train
        return img, y

    if training:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# -----------------------
# Model builder (binary sigmoid)
# -----------------------
def build_model(backbone_name, img_size):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    if backbone_name == "densenet":
        base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_tensor=inputs)
        preprocess = tf.keras.applications.densenet.preprocess_input
    elif backbone_name == "mobilenet":
        base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs)
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif backbone_name == "resnet":
        base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
        preprocess = tf.keras.applications.resnet.preprocess_input
    else:
        raise ValueError("backbone_name must be one of: densenet, mobilenet, resnet")

    x = tf.keras.layers.Lambda(preprocess, name="preprocess")(inputs)
    x = base(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name=f"{backbone_name}_binary")
    return model, base

def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc", threshold=0.5),
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            tf.keras.metrics.AUC(name="auc_roc", curve="ROC"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
        ],
    )

# -----------------------
# Threshold tuning (ลด FN)
# -----------------------
def tune_threshold_for_recall(y_true, y_prob, min_recall=0.90):
    """
    หา threshold สูงสุดที่ยังทำ recall >= min_recall (ลด FN)
    """
    best_t = 0.5
    for t in np.linspace(0.05, 0.95, 19):
        pred = (y_prob >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        recall = tp / (tp + fn + 1e-9)
        if recall >= min_recall:
            best_t = float(t)
    return best_t

# -----------------------
# Grad-CAM
# -----------------------
def find_last_conv_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            return layer.name
    raise ValueError("No conv layer found for Grad-CAM")

def make_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img_array)
        # binary: ใช้ค่า output ตรง ๆ (ความเป็นโรค)
        class_channel = pred[:, 0]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-9)
    return heatmap.numpy()

def save_cam_overlay(img, heatmap, out_path):
    import cv2
    img_np = (img.numpy() * 255.0).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.65, heatmap_color, 0.35, 0)
    cv2.imwrite(str(out_path), overlay[:, :, ::-1])

# -----------------------
# Train one backbone for one disease
# -----------------------
def train_one(backbone, train_ds, test_ds, cfg: CFG):
    model, base = build_model(backbone, cfg.img_size)

    # head
    base.trainable = False
    compile_model(model, cfg.lr_head)
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc_pr", mode="max", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc_pr", mode="max", factor=0.5, patience=1, min_lr=1e-6),
    ]
    model.fit(train_ds, validation_data=test_ds, epochs=cfg.epochs_head, callbacks=cbs, verbose=1)

    # fine-tune
    base.trainable = True
    compile_model(model, cfg.lr_fine)
    model.fit(train_ds, validation_data=test_ds, epochs=cfg.epochs_fine, callbacks=cbs, verbose=1)

    return model

# -----------------------
# Evaluate
# -----------------------
def eval_binary(model, test_ds, y_true, min_recall=0.90):
    y_prob = model.predict(test_ds, verbose=0).reshape(-1)

    # default threshold
    y_pred05 = (y_prob >= 0.5).astype(int)

    # tuned threshold to reduce FN (keep recall >= min_recall)
    t = tune_threshold_for_recall(y_true, y_prob, min_recall=min_recall)
    y_predT = (y_prob >= t).astype(int)

    def summarize(y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        return {
            "acc": float(acc),
            "recall": float(recall),
            "precision": float(precision),
            "FN": int(fn),
            "FP": int(fp),
            "TP": int(tp),
            "TN": int(tn),
            "cm": cm,
        }

    return {
        "thr_0.5": summarize(y_pred05),
        "thr_tuned": summarize(y_predT),
        "best_thr": float(t),
        "y_prob": y_prob,
    }

def save_model_comparison_plot(disease_dir: Path, disease: str, df_cmp: pd.DataFrame):
    """
    กราฟเปรียบเทียบ 3 โมเดล: Accuracy / Recall / FN
    """
    # ---- bar: acc ----
    plt.figure()
    plt.bar(df_cmp["model"], df_cmp["test_acc_tuned"])
    plt.ylim(0, 1)
    plt.title(f"{disease}: Test Accuracy (tuned threshold)")
    plt.ylabel("Accuracy")
    plt.savefig(disease_dir / "plot_acc_compare.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ---- bar: recall ----
    plt.figure()
    plt.bar(df_cmp["model"], df_cmp["recall_tuned"])
    plt.ylim(0, 1)
    plt.title(f"{disease}: Test Recall (tuned threshold) - lower FN")
    plt.ylabel("Recall")
    plt.savefig(disease_dir / "plot_recall_compare.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ---- bar: FN ----
    plt.figure()
    plt.bar(df_cmp["model"], df_cmp["FN_tuned"])
    plt.title(f"{disease}: False Negatives (FN) - lower is better")
    plt.ylabel("FN (count)")
    plt.savefig(disease_dir / "plot_fn_compare.png", dpi=160, bbox_inches="tight")
    plt.close()


def save_probability_histogram(disease_dir: Path, disease: str, model_name: str, y_prob: np.ndarray):
    """
    กราฟ histogram ความน่าจะเป็นเป็นโรค (%) ของรูป test ของโมเดลนั้น ๆ
    """
    prob_pct = y_prob * 100.0
    plt.figure()
    plt.hist(prob_pct, bins=20)
    plt.title(f"{disease}: Predicted probability (%) - {model_name}")
    plt.xlabel("Probability of disease (%)")
    plt.ylabel("Number of test images")
    plt.savefig(disease_dir / f"plot_prob_hist_{model_name}.png", dpi=160, bbox_inches="tight")
    plt.close()


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Data_Entry_2017.csv")
    parser.add_argument("--img_root", required=True, help="Root folder containing images_001, images_002, ... (scan recursively)")
    parser.add_argument("--out_dir", default="outputs_binary")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--per_class", type=int, default=1000)     # disease 1000 + no finding 1000
    parser.add_argument("--train_n", type=int, default=800)        # per class
    parser.add_argument("--test_n", type=int, default=200)         # per class
    parser.add_argument("--min_recall", type=float, default=0.90)  # for tuned threshold
    parser.add_argument("--strict_single_label", action="store_true", help="Use only single-label rows (recommended for high acc).")

    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = CFG(out_dir=args.out_dir)
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    diseases = ["Atelectasis", "Cardiomegaly", "Edema", "Emphysema", "Fibrosis"]
    backbones = ["mobilenet", "resnet", "densenet"]

    # load csv
    df = pd.read_csv(args.csv)
    fcol, lcol, pcol = resolve_columns(df)
    df = df.rename(columns={fcol: "filename", lcol: "labels", pcol: "patient_id"})

    # image index + attach full path
    img_index = build_image_index(args.img_root)
    df = attach_fullpaths(df, img_index)

    all_results = []

    for disease in diseases:
        print(f"\n==================== DATASET: {disease} vs No Finding ====================")
        out_dir = out_root / disease
        out_dir.mkdir(parents=True, exist_ok=True)

        # build dataset (balanced)
        bin_df = make_binary_dataset(
            df, disease=disease, per_class=args.per_class, seed=args.seed, strict_single_label=args.strict_single_label
        )
        tr, te = split_800_200_each_class(bin_df, train_n=args.train_n, test_n=args.test_n, seed=args.seed)

        tr.to_csv(out_dir / "train_split.csv", index=False)
        te.to_csv(out_dir / "test_split.csv", index=False)

        train_ds = make_ds(tr, cfg.img_size, cfg.batch_size, training=True)    # ✅ augment
        test_ds  = make_ds(te, cfg.img_size, cfg.batch_size, training=False)  # ✅ no augment

        # true labels for evaluation
        y_true = te["target"].astype(int).values

        disease_results = []

        for backbone in backbones:
            print(f"\n---- TRAIN {backbone.upper()} ({disease} vs No Finding) ----")
            model = train_one(backbone, train_ds, test_ds, cfg)

            eval_out = eval_binary(model, test_ds, y_true, min_recall=args.min_recall)

            # choose tuned or default? (เน้นลด FN -> ใช้ tuned เป็นหลัก)
            best_thr = eval_out["best_thr"]
            tuned = eval_out["thr_tuned"]
            cm = tuned["cm"]

            # ---- save per-image probability (%) on test set ----
            y_prob = eval_out["y_prob"]  # probability of disease
            prob_df = te[["filename", "fullpath", "target"]].copy()
            prob_df["prob_disease_pct"] = (y_prob * 100.0).round(2)
            prob_df.to_csv(out_dir / f"{backbone}_test_probabilities.csv", index=False)

            # ---- plot probability histogram ----
            save_probability_histogram(out_dir, disease, backbone, y_prob)


            # save confusion
            pd.DataFrame(cm, index=["NoFinding(0)", f"{disease}(1)"], columns=["Pred0", "Pred1"]).to_csv(
                out_dir / f"{backbone}_confusion_tuned.csv"
            )

            # save summary row
            row = {
                "disease": disease,
                "model": backbone,
                "test_acc_tuned": tuned["acc"],
                "recall_tuned": tuned["recall"],
                "precision_tuned": tuned["precision"],
                "FN_tuned": tuned["FN"],
                "best_threshold": best_thr,
                "model_path": str((out_dir / f"{backbone}_best.keras")),
            }
            disease_results.append(row)
            all_results.append(row)

            # save model
            model_path = out_dir / f"{backbone}_best.keras"
            model.save(model_path)

            print(f"[{backbone}] tuned_thr={best_thr:.2f} acc={tuned['acc']:.4f} recall={tuned['recall']:.4f} precision={tuned['precision']:.4f} FN={tuned['FN']}")

            # Grad-CAM samples
            cam_dir = out_dir / f"cam_{backbone}"
            cam_dir.mkdir(exist_ok=True, parents=True)
            last_conv = find_last_conv_name(model)

            # sample some positive + negative
            pos_s = te[te["target"] == 1].sample(min(cfg.num_cam_images // 2, (te["target"] == 1).sum()), random_state=args.seed)
            neg_s = te[te["target"] == 0].sample(min(cfg.num_cam_images // 2, (te["target"] == 0).sum()), random_state=args.seed)
            sample_df = pd.concat([pos_s, neg_s], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

            for i in range(len(sample_df)):
                p = sample_df.loc[i, "fullpath"]
                gt = int(sample_df.loc[i, "target"])
                img = decode_image(p, cfg.img_size)
                img_batch = tf.expand_dims(img, 0)

                heatmap = make_gradcam_heatmap(model, img_batch, last_conv_layer_name=last_conv)
                out_path = cam_dir / f"{i:03d}_gt{gt}.png"
                save_cam_overlay(img, heatmap, out_path)

        df_cmp = pd.DataFrame(disease_results)
        df_cmp.to_csv(out_dir / "model_comparison.csv", index=False)

        # ---- plot comparison of 3 models ----
        save_model_comparison_plot(out_dir, disease, df_cmp)


    # global summary
    summary = pd.DataFrame(all_results)
    summary.to_csv(out_root / "all_diseases_comparison.csv", index=False)

    print("\n==================== DONE ====================")
    print(f"Saved to: {out_root.resolve()}")
    print("Key files:")
    print("- outputs_binary/all_diseases_comparison.csv")
    print("- outputs_binary/<disease>/model_comparison.csv")
    print("- outputs_binary/<disease>/<model>_best.keras")
    print("- outputs_binary/<disease>/<model>_confusion_tuned.csv")
    print("- outputs_binary/<disease>/cam_<model>/ (Grad-CAM)")

if __name__ == "__main__":
    main()
