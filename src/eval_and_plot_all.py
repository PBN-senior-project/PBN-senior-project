import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# -----------------------------
# Common utils (lightweight)
# -----------------------------
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

def decode_image(path: tf.Tensor, img_size: int) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, (img_size, img_size), antialias=True)
    return img

def make_dataset(paths: np.ndarray, y: np.ndarray, img_size: int, batch_size: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, y))

    def _map_fn(x, label):
        img = decode_image(x, img_size)
        return img, tf.cast(label, tf.float32)

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# -----------------------------
# Load val split (match training logic)
# -----------------------------
def build_val_from_resolved_cfg(cfg: dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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

    # train_list filter (เหมือน train.py)
    train_list = cfg.get("train_list", None)
    if train_list and Path(train_list).exists():
        train_names = set(load_list_file(train_list))
        df_train = df[df["Image Index"].isin(train_names)].copy()
    else:
        df_train = df.copy()

    # ทำให้ split stable มากขึ้น
    df_train = df_train.sort_values("Image Index").reset_index(drop=True)

    train_df, val_df = train_test_split(
        df_train,
        test_size=float(cfg.get("val_size", 0.2)),
        random_state=int(cfg.get("seed", 42)),
        shuffle=True,
    )

    X_val = val_df["path"].values.astype(str)
    y_val = np.stack([labels_to_multihot(x, classes) for x in val_df["Finding Labels"].tolist()], axis=0)
    return X_val, y_val, classes


# -----------------------------
# Plot model graph (requires graphviz+pydot)
# -----------------------------
def try_plot_model_png(model: keras.Model, out_png: Path):
    try:
        keras.utils.plot_model(model, to_file=str(out_png), show_shapes=True, expand_nested=True, dpi=120)
        return True
    except Exception as e:
        print(f"[WARN] plot_model failed for {out_png.name}: {e}")
        return False


# -----------------------------
# Plot training history
# -----------------------------
def plot_history(history_csv: Path, out_png: Path):
    if not history_csv.exists():
        print("[WARN] history.csv not found:", history_csv)
        return

    dfh = pd.read_csv(history_csv)
    # plot whatever exists: acc/val_acc/auc/val_auc/loss/val_loss
    cols = dfh.columns.tolist()

    plt.figure()
    if "acc" in cols: plt.plot(dfh["acc"], label="acc")
    if "val_acc" in cols: plt.plot(dfh["val_acc"], label="val_acc")
    if "auc" in cols: plt.plot(dfh["auc"], label="auc")
    if "val_auc" in cols: plt.plot(dfh["val_auc"], label="val_auc")
    if "loss" in cols: plt.plot(dfh["loss"], label="loss")
    if "val_loss" in cols: plt.plot(dfh["val_loss"], label="val_loss")
    plt.title("Training History")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------
# Evaluate (acc + auc) on val
# -----------------------------
def evaluate_model_on_val(model_path: Path, cfg: dict) -> dict:
    model = keras.models.load_model(str(model_path), compile=False)

    X_val, y_val, classes = build_val_from_resolved_cfg(cfg)
    val_ds = make_dataset(X_val, y_val, int(cfg["img_size"]), int(cfg["batch_size"]))

    # compile for metrics only (loss can be generic)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.AUC(name="auc", multi_label=True, num_labels=len(classes)),
        ]
    )

    results = model.evaluate(val_ds, verbose=1)
    names = model.metrics_names  # includes 'loss', 'acc', 'auc'
    out = {names[i]: float(results[i]) for i in range(len(names))}
    out["n_val"] = int(len(X_val))
    return out


def main():
    # ✅ ใส่ชื่อโฟลเดอร์ run ของทั้งสามโมเดลตรงนี้
    # ถ้ายังมีแค่ baseline_run ก็ใส่อันเดียวก่อน แล้วค่อยเพิ่มอีกสองอัน
    run_names = [
        # "densenet_run",
        # "resnet_run",
        # "mobilenet_run",
        "baseline_run",
    ]

    runs_root = project_root() / "runs"
    report_rows = []

    for rn in run_names:
        run_dir = runs_root / rn
        cfg_path = run_dir / "resolved_config.json"
        best_path = run_dir / "best.keras"
        hist_path = run_dir / "history.csv"

        if not run_dir.exists():
            print("[SKIP] missing run_dir:", run_dir)
            continue
        if not cfg_path.exists() or not best_path.exists():
            print("[SKIP] missing cfg/model in:", run_dir)
            continue

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        out_dir = run_dir / "evaluation_outputs"
        ensure_dir(out_dir)

        # 1) plot model graph
        model = keras.models.load_model(str(best_path), compile=False)
        ok = try_plot_model_png(model, out_dir / "model_graph.png")
        if ok:
            print("[OK] saved model graph:", out_dir / "model_graph.png")
        else:
            # fallback: save model summary to txt
            summary_txt = out_dir / "model_summary.txt"
            with open(summary_txt, "w", encoding="utf-8") as f:
                model.summary(print_fn=lambda s: f.write(s + "\n"))
            print("[OK] saved model summary:", summary_txt)

        # 2) plot history
        plot_history(hist_path, out_dir / "history_plot.png")
        print("[OK] saved history plot:", out_dir / "history_plot.png")

        # 3) evaluate acc/auc
        metrics = evaluate_model_on_val(best_path, cfg)
        metrics["run_name"] = rn
        metrics["backbone"] = cfg.get("backbone", "unknown")
        report_rows.append(metrics)

        # save per-run json
        with open(out_dir / "val_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print("[OK] saved metrics:", out_dir / "val_metrics.json")

    # summary csv
    if report_rows:
        df = pd.DataFrame(report_rows)
        out_csv = runs_root / "ALL_RUNS_val_metrics.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print("[OK] wrote:", out_csv)
        print(df)
    else:
        print("No runs evaluated. Check run_names and file structure.")


if __name__ == "__main__":
    main()
