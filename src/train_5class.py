import argparse
import random
from dataclasses import dataclass
from pathlib import Path

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
    epochs_head: int = 3
    epochs_fine: int = 5
    lr_head: float = 1e-4
    lr_fine: float = 3e-5
    out_dir: str = "outputs_5class"

# -----------------------
# NIH columns
# -----------------------
COMMON_FILENAME_COLS = ["Image Index", "filename", "file"]
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
    if fcol is None or lcol is None or pcol is None:
        raise ValueError(
            "Cannot resolve columns. Found columns:\n- " + "\n- ".join(df.columns.astype(str).tolist())
        )
    return fcol, lcol, pcol

# -----------------------
# Image indexing (multi folders)
# -----------------------
def build_image_index(img_root: str, exts=(".png", ".jpg", ".jpeg")):
    img_root = Path(img_root)
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
        print(f"[warn] missing {missing:,} images -> dropping")
        print("[warn] sample:", df.loc[df["fullpath"].isna(), "filename"].head(10).tolist())
        df = df.dropna(subset=["fullpath"]).reset_index(drop=True)
    return df

# -----------------------
# Label parsing (single-label for 5-class)
# -----------------------
def parse_labels(s):
    if not isinstance(s, str) or not s.strip():
        return []
    return [x.strip() for x in s.split("|") if x.strip()]

def make_5class_df(df, classes, per_class=1000, seed=42, allow_multilabel=True):
    """
    Build 5-class dataset.
    - If allow_multilabel=True: assign each row to ONE class using priority order in `classes`.
      Example labels: "Edema|Cardiomegaly" -> assigned to "Edema" if Edema appears first in classes.
    - If allow_multilabel=False: keep only rows with exactly one label.
    """
    df = df.copy()
    df["labels_list"] = df["labels"].apply(parse_labels)

    if allow_multilabel:
        priority = {c:i for i,c in enumerate(classes)}

        def pick_primary(lbls):
            hit = [c for c in lbls if c in priority]
            if not hit:
                return None
            # choose the highest priority (smallest index)
            hit.sort(key=lambda x: priority[x])
            return hit[0]

        df["single_label"] = df["labels_list"].apply(pick_primary)
        df = df[df["single_label"].notna()].copy()
    else:
        df = df[df["labels_list"].apply(lambda x: len(x) == 1)].copy()
        df["single_label"] = df["labels_list"].apply(lambda x: x[0])
        df = df[df["single_label"].isin(classes)].copy()

    # balance
    out_parts = []
    for c in classes:
        part = df[df["single_label"] == c]
        if len(part) < per_class:
            raise ValueError(f"Class '{c}' has only {len(part)} images (<{per_class}).")
        out_parts.append(part.sample(per_class, random_state=seed))
    out = pd.concat(out_parts, ignore_index=True)

    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def split_800_200_per_class(df, classes, train_n=800, test_n=200, seed=42):
    """
    Split per class exactly 800 train / 200 test.
    (Random split. ถ้าคุณอยาก split แบบกัน patient leakage เดี๋ยวผมปรับเพิ่มได้)
    """
    tr_parts, te_parts = [], []
    for c in classes:
        part = df[df["single_label"] == c].sample(frac=1.0, random_state=seed).reset_index(drop=True)
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
    img = tf.image.resize(img, (img_size, img_size))
    return img

def augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.08)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    return img

def make_ds(df, class2idx, img_size, batch_size, training):
    paths = df["fullpath"].astype(str).values
    y = df["single_label"].map(class2idx).astype(int).values

    ds = tf.data.Dataset.from_tensor_slices((paths, y))

    def _load(p, y):
        img = decode_image(p, img_size)
        if training:
            img = augment(img)   # ✅ train only
        return img, y

    if training:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# -----------------------
# Model builder (5-class softmax)
# -----------------------
def build_model(backbone_name, num_classes, img_size):
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
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name=f"{backbone_name}_5class")
    return model, base

def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
        ],
    )

# -----------------------
# Train 1 model
# -----------------------
def train_one(backbone, train_ds, test_ds, num_classes, cfg: CFG):
    model, base = build_model(backbone, num_classes, cfg.img_size)

    # head
    base.trainable = False
    compile_model(model, cfg.lr_head)
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.5, patience=1, min_lr=1e-6),
    ]
    model.fit(train_ds, validation_data=test_ds, epochs=cfg.epochs_head, callbacks=cbs, verbose=1)

    # fine-tune
    base.trainable = True
    compile_model(model, cfg.lr_fine)
    model.fit(train_ds, validation_data=test_ds, epochs=cfg.epochs_fine, callbacks=cbs, verbose=1)

    return model

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Data_Entry_2017.csv")
    parser.add_argument("--img_root", required=True, help="Root that contains images_001, images_002, ...")
    parser.add_argument("--out_dir", default="outputs_5class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_class", type=int, default=1000)  # ✅ 1000/class
    parser.add_argument("--train_n", type=int, default=800)     # ✅ 800
    parser.add_argument("--test_n", type=int, default=200)      # ✅ 200
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = CFG(out_dir=args.out_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # target 5 diseases
    classes = ["Atelectasis", "Cardiomegaly", "Edema", "Emphysema", "Fibrosis"]
    class2idx = {c:i for i,c in enumerate(classes)}
    idx2class = {i:c for c,i in class2idx.items()}

    # load csv + normalize column names
    df = pd.read_csv(args.csv)
    fcol, lcol, pcol = resolve_columns(df)
    df = df.rename(columns={fcol:"filename", lcol:"labels", pcol:"patient_id"})

    # index images + attach fullpath
    img_index = build_image_index(args.img_root)
    df = attach_fullpaths(df, img_index)

    # build balanced 5-class dataset (single-label only)
    df5 = make_5class_df(df, classes, per_class=args.per_class, seed=args.seed, allow_multilabel=True)
    tr, te = split_800_200_per_class(df5, classes, train_n=args.train_n, test_n=args.test_n, seed=args.seed)

    # save splits
    tr.to_csv(out_dir / "train_split.csv", index=False)
    te.to_csv(out_dir / "test_split.csv", index=False)

    train_ds = make_ds(tr, class2idx, cfg.img_size, cfg.batch_size, training=True)   # ✅ augment
    test_ds  = make_ds(te, class2idx, cfg.img_size, cfg.batch_size, training=False) # ✅ no augment

    results = []
    for backbone in ["mobilenet", "resnet", "densenet"]:
        print(f"\n==================== TRAIN {backbone.upper()} ====================")
        model = train_one(backbone, train_ds, test_ds, num_classes=len(classes), cfg=cfg)

        # evaluate + report
        probs = model.predict(test_ds, verbose=0)
        y_pred = probs.argmax(axis=1)

        # true labels from te
        y_true = te["single_label"].map(class2idx).astype(int).values

        acc = float((y_pred == y_true).mean())

        # confusion + per-class FN
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        # FN for class i = sum row i except diagonal
        fn_per_class = cm.sum(axis=1) - np.diag(cm)
        total_fn = int(fn_per_class.sum())

        rep = classification_report(
            y_true, y_pred,
            target_names=classes,
            zero_division=0,
            output_dict=True
        )
        macro_recall = float(rep["macro avg"]["recall"])
        macro_f1 = float(rep["macro avg"]["f1-score"])

        # save artifacts
        model_path = out_dir / f"{backbone}_best.keras"
        model.save(model_path)

        pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_dir / f"{backbone}_confusion.csv")
        pd.DataFrame({"class": classes, "FN": fn_per_class}).to_csv(out_dir / f"{backbone}_fn_per_class.csv", index=False)

        results.append({
            "model": backbone,
            "test_acc": acc,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "total_FN": total_fn,
            "model_path": str(model_path),
        })

        print(f"[{backbone}] acc={acc:.4f} macro_recall={macro_recall:.4f} macro_f1={macro_f1:.4f} total_FN={total_fn}")
        print("FN per class:", dict(zip(classes, fn_per_class.tolist())))

    res_df = pd.DataFrame(results).sort_values(
        by=["test_acc", "total_FN"], ascending=[False, True]
    )
    res_df.to_csv(out_dir / "model_comparison.csv", index=False)

    print("\n==================== FINAL RANKING ====================")
    print(res_df)
    print(f"\nSaved to: {out_dir.resolve()}")
    print("Key files:")
    print("- model_comparison.csv")
    print("- <model>_confusion.csv")
    print("- <model>_fn_per_class.csv")
    print("- train_split.csv / test_split.csv")

if __name__ == "__main__":
    main()
