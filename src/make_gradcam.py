from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


CLASSES = [
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Nodule",
    "Mass",
    "Pneumothorax",
]

IMG_SIZE = 384
SEED = 42
N_NO_FINDING = 5000


def root():
    return Path(__file__).resolve().parents[1]


def find_images(folder):
    return {
        p.name: str(p.resolve())
        for p in folder.rglob("*.png")
    }


def apply_clahe(img):
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


def load_image(path):
    img = cv2.imread(str(path))

    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # original image for display
    original = img.astype(np.float32) / 255.0

    # same validation preprocessing as train_V7.py
    processed = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    processed = apply_clahe(processed)
    processed = processed.astype(np.float32) / 255.0

    return original, processed


def build_validation(csv_path, img_root):
    print("[1] Building validation set...")

    df = pd.read_csv(csv_path)
    img_map = find_images(img_root)

    df["filepath"] = df["Image Index"].map(img_map)
    df = df[df["filepath"].notna()].copy()

    # PA/AP only
    df["View Position"] = (
        df["View Position"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["View Position"].isin(["PA", "AP"])
    ].copy()

    # labels
    labels = df["Finding Labels"].fillna("").str.split("|")

    for cls in CLASSES:
        df[cls] = labels.apply(
            lambda x: int(cls in x)
        )

    # positive 6 findings
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

    # patient-level split
    patients = selected["Patient ID"].unique()

    train_patients, val_patients = train_test_split(
        patients,
        test_size=0.2,
        random_state=SEED
    )

    val_df = selected[
        selected["Patient ID"].isin(val_patients)
    ].copy().reset_index(drop=True)

    print(f"Selected   : {len(selected)}")
    print(f"Validation : {len(val_df)}")
    print(val_df["View Position"].value_counts())

    return val_df


def find_last_conv(model):
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name

    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, keras.layers.Conv2D):
                    return sub.name

    raise ValueError("Conv2D layer not found")


def make_gradcam(model, image, class_idx, conv_name):
    conv_layer = model.get_layer(conv_name)

    grad_model = keras.Model(
        model.inputs,
        [conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(
            image,
            training=False
        )

        score = pred[:, class_idx]

    grads = tape.gradient(
        score,
        conv_out
    )

    weights = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_out = conv_out[0]

    heatmap = tf.reduce_sum(
        conv_out * weights,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap /= (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


def save_gradcam(
    original,
    heatmap,
    path,
    title
):
    heatmap = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(10, 5)
    )

    ax[0].imshow(original, cmap="gray")
    ax[0].set_title("Original X-ray")
    ax[0].axis("off")

    ax[1].imshow(original, cmap="gray")
    ax[1].imshow(
        heatmap,
        cmap="jet",
        alpha=0.40
    )
    ax[1].set_title(title)
    ax[1].axis("off")

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


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

    img_root = r / "archive"

    out_dir = (
        r / "outputs"
        / "gradcam_v7"
        / "DenseNet121"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------
    # validation set
    # ---------------------------------
    val_df = build_validation(
        csv_path,
        img_root
    )

    # ---------------------------------
    # model
    # ---------------------------------
    print("\n[2] Loading DenseNet121...")

    model = keras.models.load_model(
        model_path,
        compile=False
    )

    conv_name = find_last_conv(model)

    print(
        "Last Conv Layer:",
        conv_name
    )

    # ---------------------------------
    # one positive example per disease
    # ---------------------------------
    for disease in CLASSES:
        cases = val_df[
            val_df[disease] == 1
        ]

        if cases.empty:
            print(
                f"[SKIP] No validation case: "
                f"{disease}"
            )
            continue

        # reproducible sample
        row = cases.sample(
            1,
            random_state=SEED
        ).iloc[0]

        original, processed = load_image(
            row["filepath"]
        )

        inp = tf.convert_to_tensor(
            processed[None, ...],
            dtype=tf.float32
        )

        pred = model.predict(
            inp,
            verbose=0
        )[0]

        class_idx = CLASSES.index(
            disease
        )

        score = float(
            pred[class_idx]
        )

        heatmap = make_gradcam(
            model,
            inp,
            class_idx,
            conv_name
        )

        view = row["View Position"]
        image_name = row["Image Index"]

        title = (
            f"{disease} | "
            f"{view} | "
            f"Score={score:.3f}"
        )

        out_path = (
            out_dir
            / f"GradCAM_{disease}_{view}_{image_name}.png"
        )

        save_gradcam(
            original,
            heatmap,
            out_path,
            title
        )

        print(
            f"[OK] {disease:15} "
            f"{view} "
            f"score={score:.3f}"
        )

    print(
        "\nSaved to:",
        out_dir
    )


if __name__ == "__main__":
    main()