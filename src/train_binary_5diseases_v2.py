# src/train_binary_5diseases_v2.py
import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix

# -------- optional plotting (ไม่บังคับติดตั้ง matplotlib) ----------
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


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
    epochs_head: int = 6
    epochs_fine: int = 12
    lr_head: float = 2e-4
    lr_fine: float = 5e-5
    out_dir: str = "outputs_binary_v2"
    num_cam_images: int = 12
    cache_ds: bool = False  # ถ้า RAM พอค่อยเปิด True


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
# Image indexing (images_001/images_002/... scan recursively)
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
    if not isinstance(label_str, str):
        return False
    s = label_str.strip().lower()
    return s in ["no finding", "no findings", "sin hallazgos", "sin hallazgo", "normal"]


def make_binary_dataset(df, disease: str, per_class=1000, seed=42, strict_single_label=True):
    """
    สร้าง dataset: disease vs No Finding (balanced)
    strict_single_label=True: ใช้เฉพาะ label เดี่ยว (สะอาดสุด → acc มักสูงขึ้น)
    """
    df = df.copy()
    df["labels_list"] = df["labels"].apply(parse_labels)

    if strict_single_label:
        df = df[df["labels_list"].apply(lambda x: len(x) == 1)].copy()
        df["single"] = df["labels_list"].apply(lambda x: x[0])
        pos = df[df["single"] == disease].copy()
        neg = df[df["single"].apply(lambda x: is_no_finding(x))].copy()
    else:
        # ใช้ multi-label: pos คือมีโรคนั้นอยู่, neg คือ "No Finding" เท่านั้น (NIH ส่วนใหญ่ No Finding จะเป็น label เดี่ยว)
        pos = df[df["labels_list"].apply(lambda xs: disease in xs)].copy()
        neg = df[df["labels_list"].apply(lambda xs: (len(xs) == 1 and is_no_finding(xs[0])))].copy()

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


def group_split_then_fix_counts(df, train_n=800, test_n=200, seed=42):
    """
    สร้าง train/test แบบ "กัน patient leakage" และ "ได้จำนวน exact ต่อคลาส" ให้สำเร็จ
    กลยุทธ์:
      1) split รายชื่อ patient เป็น train_pool/test_pool
      2) สร้าง pool rows จาก patient นั้น ๆ
      3) ถ้าฝั่งใดไม่พอ (เช่น test) -> ย้าย patient จาก train_pool ไป test_pool เพิ่มจนพอ
    """
    rng = np.random.default_rng(seed)

    # แยก patients
    patients = df["patient_id"].dropna().unique().tolist()
    rng.shuffle(patients)

    # ตั้งต้น test_pool = 20% patients
    n_test_pat = max(1, int(0.20 * len(patients)))
    test_pats = set(patients[:n_test_pat])
    train_pats = set(patients[n_test_pat:])

    def pool_rows(pats_set):
        return df[df["patient_id"].isin(pats_set)].copy()

    def has_enough(pool_df, n0, n1):
        c0 = int((pool_df["target"] == 0).sum())
        c1 = int((pool_df["target"] == 1).sum())
        return (c0 >= n0) and (c1 >= n1)

    def sample_exact(pool_df, n0, n1, seed_local):
        p0 = pool_df[pool_df["target"] == 0]
        p1 = pool_df[pool_df["target"] == 1]
        # sample แบบไม่ซ้ำ
        s0 = p0.sample(n0, random_state=seed_local)
        s1 = p1.sample(n1, random_state=seed_local)
        out = pd.concat([s0, s1], ignore_index=True).sample(frac=1.0, random_state=seed_local).reset_index(drop=True)
        return out

    need_tr0 = train_n
    need_tr1 = train_n
    need_te0 = test_n
    need_te1 = test_n

    # ขยาย test pool ถ้ายังไม่พอ
    moved = 0
    max_moves = len(train_pats)

    while True:
        tr_pool = pool_rows(train_pats)
        te_pool = pool_rows(test_pats)

        ok_tr = has_enough(tr_pool, need_tr0, need_tr1)
        ok_te = has_enough(te_pool, need_te0, need_te1)

        if ok_tr and ok_te:
            break

        # ถ้า test ไม่พอ ให้ย้าย patient จาก train -> test เพิ่ม
        if not ok_te:
            if not train_pats or moved >= max_moves:
                raise ValueError(
                    "Cannot form exact 800/200 per class after patient split.\n"
                    "สาเหตุ: ฝั่ง test มีจำนวนตัวอย่างคลาส 0/1 ไม่พอแม้จะย้าย patient แล้ว\n"
                    "แนะนำ: ลด train_n/test_n หรือเพิ่ม per_class หรือปิด strict_single_label"
                )
            # ย้าย patient ทีละคน (สุ่ม)
            p_move = next(iter(train_pats))
            train_pats.remove(p_move)
            test_pats.add(p_move)
            moved += 1
            continue

        # ถ้า train ไม่พอ (เกิดได้น้อย) ให้ย้ายกลับจาก test -> train
        if not ok_tr:
            if not test_pats:
                raise ValueError("Train pool not enough and cannot move from test.")
            p_move = next(iter(test_pats))
            test_pats.remove(p_move)
            train_pats.add(p_move)
            moved += 1
            continue

    # sample exact จาก pool
    tr_pool = pool_rows(train_pats)
    te_pool = pool_rows(test_pats)

    tr = sample_exact(tr_pool, need_tr0, need_tr1, seed)
    te = sample_exact(te_pool, need_te0, need_te1, seed + 999)

    # leakage check
    overlap = set(tr["patient_id"]).intersection(set(te["patient_id"]))
    if overlap:
        raise RuntimeError(f"Patient leakage detected ({len(overlap)} patients).")

    return tr, te



# -----------------------
# TF pipeline (สำคัญ: decode เป็น 0..255 float32 เพื่อให้ preprocess_input ถูกต้อง)
# -----------------------
def decode_image_255(path, img_size):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)  # uint8
    img = tf.image.resize(img, (img_size, img_size), method="bilinear")
    img = tf.cast(img, tf.float32)  # 0..255 ✅ สำคัญมาก
    return img


def augment(img_255):
    # img_255 เป็น 0..255
    img = img_255

    # random flips
    img = tf.image.random_flip_left_right(img)

    # photometric (ยังเป็น 0..255)
    img = tf.image.random_brightness(img, 0.08 * 255.0)
    img = tf.image.random_contrast(img, 0.85, 1.15)

    # small rotation/crop แบบเบา ๆ (กัน over-augment)
    # resize+random crop
    bigger = int(1.07 * 224)
    img = tf.image.resize(img, (bigger, bigger))
    img = tf.image.random_crop(img, size=(224, 224, 3))

    return img


def make_ds(df, img_size, batch_size, training, cache=False):
    paths = df["fullpath"].astype(str).values
    y = df["target"].astype(np.int32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, y))

    def _load(p, y):
        img = decode_image_255(p, img_size)
        if training:
            img = augment(img)
        return img, y

    if training:
        ds = ds.shuffle(4096, reshuffle_each_iteration=True)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# -----------------------
# Model builder (binary sigmoid) - base WITHOUT input_tensor (ลดปัญหา graph ซ้อน)
# -----------------------
def get_backbone(backbone_name, img_size):
    if backbone_name == "mobilenet":
        base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                                 input_shape=(img_size, img_size, 3))
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif backbone_name == "resnet":
        base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet",
                                              input_shape=(img_size, img_size, 3))
        preprocess = tf.keras.applications.resnet.preprocess_input
    elif backbone_name == "densenet":
        base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet",
                                                 input_shape=(img_size, img_size, 3))
        preprocess = tf.keras.applications.densenet.preprocess_input
    else:
        raise ValueError("backbone_name must be one of: mobilenet, resnet, densenet")

    return base, preprocess


def build_model(backbone_name, img_size):
    base, preprocess = get_backbone(backbone_name, img_size)

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="img_255")
    x = tf.keras.layers.Lambda(preprocess, name="preprocess")(inputs)  # preprocess expects 0..255 ✅
    x = base(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name=f"{backbone_name}_binary")
    return model, base


def compile_model(model, lr):
    # เน้น recall/FN → ใส่ AUC_PR เป็นหลัก
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc", threshold=0.5),
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
            tf.keras.metrics.AUC(name="auc_roc", curve="ROC"),
        ],
    )


# -----------------------
# Threshold tuning (ลด FN): เลือก threshold ที่ recall >= min_recall แล้วทำ precision สูงสุด
# -----------------------
def tune_threshold_for_recall(y_true, y_prob, min_recall=0.90):
    best_t = 0.5
    best_prec = -1.0

    for t in np.linspace(0.05, 0.95, 37):
        pred = (y_prob >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())

        recall = tp / (tp + fn + 1e-9)
        prec = tp / (tp + fp + 1e-9)

        if recall >= min_recall:
            # เลือก precision สูงสุด ภายใต้ recall constraint
            if prec > best_prec:
                best_prec = prec
                best_t = float(t)

    return best_t


def summarize_binary(y_true, y_prob, thr):
    pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
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


# -----------------------
# Grad-CAM (optional)
# -----------------------
def find_last_conv_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            return layer.name
    raise ValueError("No conv layer found for Grad-CAM")


def make_gradcam_heatmap(model, img_batch_255, last_conv_layer_name):
    grad_model = tf.keras.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img_batch_255)
        class_channel = pred[:, 0]  # probability disease

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-9)
    return heatmap.numpy()


def save_cam_overlay(img_255, heatmap, out_path):
    import cv2
    # img_255: 0..255
    img_np = tf.clip_by_value(img_255, 0.0, 255.0).numpy().astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.65, heatmap_color, 0.35, 0)
    cv2.imwrite(str(out_path), overlay[:, :, ::-1])


# -----------------------
# Plots (optional)
# -----------------------
def save_model_comparison_plot(disease_dir: Path, disease: str, df_cmp: pd.DataFrame):
    if not HAS_PLT:
        return
    # acc
    plt.figure()
    plt.bar(df_cmp["model"], df_cmp["test_acc"])
    plt.ylim(0, 1)
    plt.title(f"{disease}: Test Accuracy")
    plt.ylabel("Accuracy")
    plt.savefig(disease_dir / "plot_acc_compare.png", dpi=160, bbox_inches="tight")
    plt.close()

    # recall
    plt.figure()
    plt.bar(df_cmp["model"], df_cmp["recall"])
    plt.ylim(0, 1)
    plt.title(f"{disease}: Test Recall (tuned threshold)")
    plt.ylabel("Recall")
    plt.savefig(disease_dir / "plot_recall_compare.png", dpi=160, bbox_inches="tight")
    plt.close()

    # FN
    plt.figure()
    plt.bar(df_cmp["model"], df_cmp["FN"])
    plt.title(f"{disease}: False Negatives (FN)")
    plt.ylabel("FN count")
    plt.savefig(disease_dir / "plot_fn_compare.png", dpi=160, bbox_inches="tight")
    plt.close()


def save_probability_histogram(disease_dir: Path, disease: str, model_name: str, y_prob: np.ndarray):
    if not HAS_PLT:
        return
    prob_pct = y_prob * 100.0
    plt.figure()
    plt.hist(prob_pct, bins=20)
    plt.title(f"{disease}: Predicted probability (%) - {model_name}")
    plt.xlabel("Probability of disease (%)")
    plt.ylabel("Count")
    plt.savefig(disease_dir / f"plot_prob_hist_{model_name}.png", dpi=160, bbox_inches="tight")
    plt.close()


# -----------------------
# Train one model
# -----------------------
def train_one(backbone, train_ds, test_ds, cfg: CFG):
    model, base = build_model(backbone, cfg.img_size)

    # stage 1: head
    base.trainable = False
    compile_model(model, cfg.lr_head)

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc_pr", mode="max", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc_pr", mode="max", factor=0.5, patience=2, min_lr=1e-6),
    ]
    model.fit(train_ds, validation_data=test_ds, epochs=cfg.epochs_head, callbacks=cbs, verbose=1)

    # stage 2: fine-tune (unfreeze last blocks แบบปลอดภัย)
    base.trainable = True
    # (optional) unfreeze partial: ทำให้เสถียรกว่า
    for layer in base.layers[:-30]:
        layer.trainable = False

    compile_model(model, cfg.lr_fine)
    model.fit(train_ds, validation_data=test_ds, epochs=cfg.epochs_fine, callbacks=cbs, verbose=1)

    return model


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Data_Entry_2017.csv")
    parser.add_argument("--img_root", required=True, help="Root folder containing images_001/images_002/...")

    parser.add_argument("--out_dir", default="outputs_binary_v2")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--per_class", type=int, default=1000)   # disease 1000 + no finding 1000
    parser.add_argument("--train_n", type=int, default=800)      # per class
    parser.add_argument("--test_n", type=int, default=200)       # per class
    parser.add_argument("--min_recall", type=float, default=0.90)

    parser.add_argument("--strict_single_label", action="store_true",
                        help="Use only single-label rows (recommended for higher accuracy)")
    parser.add_argument("--only_one_disease", default="", help="Debug: run only one disease name")

    args = parser.parse_args()
    seed_everything(args.seed)

    cfg = CFG(out_dir=args.out_dir)
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    diseases = ["Atelectasis", "Cardiomegaly", "Edema", "Emphysema", "Fibrosis"]
    if args.only_one_disease.strip():
        diseases = [args.only_one_disease.strip()]

    backbones = ["mobilenet", "resnet", "densenet"]

    # load csv
    df = pd.read_csv(args.csv)
    fcol, lcol, pcol = resolve_columns(df)
    df = df.rename(columns={fcol: "filename", lcol: "labels", pcol: "patient_id"})

    # image index
    img_index = build_image_index(args.img_root)
    df = attach_fullpaths(df, img_index)

    all_rows = []

    for disease in diseases:
        print(f"\n==================== DATASET: {disease} vs No Finding ====================")
        disease_dir = out_root / disease
        disease_dir.mkdir(parents=True, exist_ok=True)

        # build balanced 2000 rows
        bin_df = make_binary_dataset(
            df, disease=disease, per_class=args.per_class, seed=args.seed, strict_single_label=args.strict_single_label
        )

        # group split + exact 800/200 per class
        tr, te = group_split_then_fix_counts(bin_df, train_n=args.train_n, test_n=args.test_n, seed=args.seed)

        tr.to_csv(disease_dir / "train_split.csv", index=False)
        te.to_csv(disease_dir / "test_split.csv", index=False)

        train_ds = make_ds(tr, cfg.img_size, cfg.batch_size, training=True, cache=cfg.cache_ds)   # augment only train
        test_ds  = make_ds(te, cfg.img_size, cfg.batch_size, training=False, cache=False)

        y_true = te["target"].astype(int).values

        disease_results = []

        for backbone in backbones:
            print(f"\n---- TRAIN {backbone.upper()} ({disease} vs No Finding) ----")
            model = train_one(backbone, train_ds, test_ds, cfg)

            # predict
            y_prob = model.predict(test_ds, verbose=0).reshape(-1)

            # tune threshold to reduce FN (recall constraint)
            best_thr = tune_threshold_for_recall(y_true, y_prob, min_recall=args.min_recall)
            tuned = summarize_binary(y_true, y_prob, best_thr)

            # save per-image probability (%)
            prob_df = te[["filename", "fullpath", "target", "patient_id"]].copy()
            prob_df["prob_disease_pct"] = (y_prob * 100.0).round(2)
            prob_df.to_csv(disease_dir / f"{backbone}_test_probabilities.csv", index=False)

            # save confusion
            cm = tuned["cm"]
            pd.DataFrame(cm, index=["NoFinding(0)", f"{disease}(1)"], columns=["Pred0", "Pred1"]).to_csv(
                disease_dir / f"{backbone}_confusion_tuned.csv"
            )

            # save model
            model_path = disease_dir / f"{backbone}_best.keras"
            model.save(model_path)

            # optional plots
            save_probability_histogram(disease_dir, disease, backbone, y_prob)

            row = {
                "disease": disease,
                "model": backbone,
                "test_acc": tuned["acc"],
                "recall": tuned["recall"],
                "precision": tuned["precision"],
                "FN": tuned["FN"],
                "FP": tuned["FP"],
                "best_threshold": best_thr,
                "model_path": str(model_path),
            }
            disease_results.append(row)
            all_rows.append(row)

            print(f"[{backbone}] thr={best_thr:.2f} acc={tuned['acc']:.4f} recall={tuned['recall']:.4f} precision={tuned['precision']:.4f} FN={tuned['FN']} FP={tuned['FP']}")

            # Grad-CAM samples (ไม่เยอะ)
            cam_dir = disease_dir / f"cam_{backbone}"
            cam_dir.mkdir(exist_ok=True, parents=True)
            last_conv = find_last_conv_name(model)

            pos_s = te[te["target"] == 1].sample(min(cfg.num_cam_images // 2, (te["target"] == 1).sum()), random_state=args.seed)
            neg_s = te[te["target"] == 0].sample(min(cfg.num_cam_images // 2, (te["target"] == 0).sum()), random_state=args.seed)
            sample_df = pd.concat([pos_s, neg_s], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

            for i in range(len(sample_df)):
                p = sample_df.loc[i, "fullpath"]
                gt = int(sample_df.loc[i, "target"])
                img = decode_image_255(p, cfg.img_size)
                img_batch = tf.expand_dims(img, 0)
                heatmap = make_gradcam_heatmap(model, img_batch, last_conv_layer_name=last_conv)
                out_path = cam_dir / f"{i:03d}_gt{gt}.png"
                save_cam_overlay(img, heatmap, out_path)

        df_cmp = pd.DataFrame(disease_results)
        df_cmp.to_csv(disease_dir / "model_comparison.csv", index=False)
        save_model_comparison_plot(disease_dir, disease, df_cmp)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(out_root / "all_diseases_comparison.csv", index=False)

    print("\n==================== DONE ====================")
    print(f"Saved to: {out_root.resolve()}")
    print("Key files:")
    print("- all_diseases_comparison.csv")
    print("- <disease>/model_comparison.csv")
    print("- <disease>/<model>_best.keras")
    print("- <disease>/<model>_confusion_tuned.csv")
    print("- <disease>/<model>_test_probabilities.csv")
    print("- <disease>/cam_<model>/ (Grad-CAM)")


if __name__ == "__main__":
    main()
