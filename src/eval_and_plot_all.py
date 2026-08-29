import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)

CLASSES = [
    "Infiltration", "Effusion", "Atelectasis",
    "Nodule", "Mass", "Pneumothorax"
]

IMG_SIZE = 384
BATCH_SIZE = 8
SEED = 42
THRESHOLD = 0.5
N_NO_FINDING = 5000


def root():
    return Path(__file__).resolve().parents[1]


def find_images(folder):
    return {
        p.name: str(p.resolve())
        for p in folder.rglob("*.png")
    }


def preprocess_np(path):
    path = path.numpy().decode()
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(l)

    img = cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2RGB
    )

    return img.astype(np.float32) / 255.0


def preprocess_tf(path):
    img = tf.py_function(
        preprocess_np,
        [path],
        tf.float32
    )
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


def make_dataset(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(
        preprocess_tf,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# =========================================================
# BUILD SAME VALIDATION SET AS train_V7.py
# =========================================================
def build_val(csv_path, image_root):

    print("\n[1] Loading dataset...")

    df = pd.read_csv(csv_path)
    image_map = find_images(image_root)

    print(f"Images found: {len(image_map)}")

    df["filepath"] = df["Image Index"].map(image_map)

    df = df[df["filepath"].notna()].copy()

    # PA / AP only
    df["View Position"] = (
        df["View Position"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["View Position"].isin(["PA", "AP"])
    ].copy()

    # 6 labels
    labels = df["Finding Labels"].fillna("").str.split("|")

    for cls in CLASSES:
        df[cls] = labels.apply(
            lambda x: int(cls in x)
        )

    # Positive 6 findings
    positive = df[
        df[CLASSES].sum(axis=1) > 0
    ].copy()

    # No Finding max 5000
    negative = df[
        df["Finding Labels"].eq("No Finding")
    ].copy()

    negative = negative.sample(
        n=min(N_NO_FINDING, len(negative)),
        random_state=SEED
    )

    selected = pd.concat(
        [positive, negative],
        ignore_index=True
    ).sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    # Patient-level split
    patients = selected["Patient ID"].unique()

    train_patients, val_patients = train_test_split(
        patients,
        test_size=0.2,
        random_state=SEED
    )

    train_df = selected[
        selected["Patient ID"].isin(train_patients)
    ]

    val_df = selected[
        selected["Patient ID"].isin(val_patients)
    ].copy().reset_index(drop=True)

    overlap = (
        set(train_df["Patient ID"])
        & set(val_df["Patient ID"])
    )

    assert len(overlap) == 0

    print(f"Selected   : {len(selected)}")
    print(f"Train      : {len(train_df)}")
    print(f"Validation : {len(val_df)}")
    print(f"Overlap    : {len(overlap)}")

    print("\nValidation View Position:")
    print(val_df["View Position"].value_counts())

    return val_df


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def class_metrics(y_true, y_score):
    y_pred = (y_score >= THRESHOLD).astype(int)

    return {
        "AUC": safe_auc(y_true, y_score),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "Positive": int(y_true.sum()),
        "Negative": int(len(y_true) - y_true.sum())
    }


# =========================================================
# EVALUATION
# =========================================================
def evaluate(model_path, val_df, out_dir):

    print("\n[2] Loading DenseNet121...")

    model = keras.models.load_model(
        model_path,
        compile=False
    )

    x = val_df["filepath"].values.astype(str)
    y_true = val_df[CLASSES].values.astype(int)

    print("\n[3] Predicting validation set...")

    y_score = model.predict(
        make_dataset(x),
        verbose=1
    )[:len(y_true)]

    y_pred = (y_score >= THRESHOLD).astype(int)

    # ---------------- Overall ----------------
    aucs = [
        safe_auc(y_true[:, i], y_score[:, i])
        for i in range(len(CLASSES))
    ]

    overall = {
        "Model": "DenseNet121",
        "Validation Images": len(y_true),
        "Threshold": THRESHOLD,
        "Accuracy": float(np.mean(y_true == y_pred)),
        "Precision Micro": precision_score(
            y_true, y_pred,
            average="micro",
            zero_division=0
        ),
        "Recall Micro": recall_score(
            y_true, y_pred,
            average="micro",
            zero_division=0
        ),
        "F1 Micro": f1_score(
            y_true, y_pred,
            average="micro",
            zero_division=0
        ),
        "Precision Macro": precision_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),
        "Recall Macro": recall_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),
        "F1 Macro": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),
        "Macro AUC": float(np.nanmean(aucs))
    }

    print("\n" + "=" * 65)
    print("DENSENET121 OVERALL RESULTS")
    print("=" * 65)

    for k, v in overall.items():
        if isinstance(v, float):
            print(f"{k:20}: {v:.4f}")
        else:
            print(f"{k:20}: {v}")

    # ---------------- PA / AP ----------------
    rows = []

    for view in ["PA", "AP"]:

        mask = val_df["View Position"].values == view
        yt = y_true[mask]
        ys = y_score[mask]

        print("\n" + "=" * 80)
        print(f"{view} - 6 FINDINGS | Images = {len(yt)}")
        print("=" * 80)
        print(
            f"{'Disease':15} {'AUC':>8} {'ACC':>8} "
            f"{'Precision':>10} {'Recall':>8} {'F1':>8}"
        )
        print("-" * 80)

        for i, disease in enumerate(CLASSES):

            m = class_metrics(
                yt[:, i],
                ys[:, i]
            )

            rows.append({
                "Disease": disease,
                "View": view,
                "N": len(yt),
                **m
            })

            print(
                f"{disease:15} "
                f"{m['AUC']:8.4f} "
                f"{m['Accuracy']:8.4f} "
                f"{m['Precision']:10.4f} "
                f"{m['Recall']:8.4f} "
                f"{m['F1']:8.4f}"
            )

    # ---------------- Save ----------------
    result_df = pd.DataFrame(rows)

    with open(
        out_dir / "DenseNet121_Overall.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            overall,
            f,
            indent=2,
            ensure_ascii=False
        )

    result_df.to_csv(
        out_dir / "DenseNet121_AP_PA.csv",
        index=False,
        encoding="utf-8-sig"
    )

    result_df[
        result_df["View"] == "PA"
    ].to_csv(
        out_dir / "DenseNet121_PA.csv",
        index=False,
        encoding="utf-8-sig"
    )

    result_df[
        result_df["View"] == "AP"
    ].to_csv(
        out_dir / "DenseNet121_AP.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[OK] Results saved to:")
    print(out_dir)


# =========================================================
# MAIN
# =========================================================
def main():

    r = root()

    model_path = (
        r / "models_v7"
        / "best_DenseNet121_v7.keras"
    )

    csv_path = (
        r / "archive"
        / "Data_Entry_2017.csv"
    )

    image_root = r / "archive"

    out_dir = (
        r / "outputs"
        / "evaluation_v7"
        / "DenseNet121"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    print("=" * 65)
    print("PBN CHEST X-RAY - DENSENET121 EVALUATION")
    print("=" * 65)

    val_df = build_val(
        csv_path,
        image_root
    )

    val_df.to_csv(
        out_dir / "validation_cases.csv",
        index=False,
        encoding="utf-8-sig"
    )

    evaluate(
        model_path,
        val_df,
        out_dir
    )


if __name__ == "__main__":
    main()