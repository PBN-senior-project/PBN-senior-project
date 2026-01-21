# train.py
# ============================================================
# Multilabel CXR (NIH) Training Script
# - Backbones: MobileNetV2 / ResNet50 / DenseNet121
# - Handles imbalance using: per-class pos_weight + Weighted Focal Loss
# - Two-stage training: (1) train head (AdamW) (2) fine-tune last 20% (SGD+momentum)
# - Metrics: PR-AUC / ROC-AUC (multi-label)
# - Eval: tune per-class thresholds on VAL (maximize F2), then test metrics + confusions
# - Saves: models (*.keras), plots, reports in outputs/runs_multilabel
# ============================================================

import os
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# -------------------------
# Config
# -------------------------
@dataclass
class CFG:
    # ROOT = project root (e.g. D:\senior-project)
    ROOT: Path = Path(__file__).resolve().parents[1]

    NIH_DIR: Path = ROOT / "data" / "nih"
    CSV_PATH: Path = NIH_DIR / "Data_Entry_2017.csv"

    IMG_GLOB: str = "images_*/images/*.png"

    TARGETS = ["Atelectasis", "Cardiomegaly", "Edema", "Emphysema", "Fibrosis"]
    SEED: int = 42

    IMG_SIZE = (224, 224)
    BATCH = 16

    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.2

    E_HEAD = 5
    E_FINE = 5
    LR1 = 1e-4
    LR2 = 3e-5

    UNFREEZE_PCT = 0.2
    KEEP_BN_FROZEN = True

    FOCAL_GAMMA = 2.0

    RUNS_DIR: Path = ROOT / "outputs" / "runs_multilabel"


cfg = CFG()
cfg.RUNS_DIR.mkdir(parents=True, exist_ok=True)

print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

seed_everything(cfg.SEED)
print("ROOT:", cfg.ROOT)
print("CSV exists:", cfg.CSV_PATH.exists(), cfg.CSV_PATH)


# ============================================================
# Data preparation
# ============================================================
def index_image_paths(nih_dir: Path, pattern: str):
    paths = list(nih_dir.glob(pattern))
    mp = {p.name: str(p) for p in paths}
    print("[INFO] Indexed images:", len(mp))
    return mp

def build_dataframe(cfg: CFG) -> pd.DataFrame:
    print("[INFO] Loading CSV:", cfg.CSV_PATH)
    df = pd.read_csv(cfg.CSV_PATH)

    # Map file name -> path
    mp = index_image_paths(cfg.NIH_DIR, cfg.IMG_GLOB)
    df["path"] = df["Image Index"].map(mp.get)
    df = df.dropna(subset=["path"]).reset_index(drop=True)

    # Multi-label columns for targets
    lbl = df["Finding Labels"].astype(str)
    for t in cfg.TARGETS:
        df[t] = lbl.str.contains(t, regex=False).astype("float32")

    # NIH "No Finding" as all-zero (for these targets)
    df["is_nofinding"] = (lbl == "No Finding").astype("int")

    # Keep rows that are either: any of 5 diseases OR "No Finding"
    any5 = df[cfg.TARGETS].sum(axis=1) > 0
    df = df[any5 | (df["is_nofinding"] == 1)].reset_index(drop=True)

    print("[INFO] Rows kept:", len(df))
    print("[INFO] No Finding rows:", int(df["is_nofinding"].sum()))
    print("[INFO] Pos counts:", df[cfg.TARGETS].sum().astype(int).to_dict())
    return df

def group_split(df: pd.DataFrame, cfg: CFG):
    gss = GroupShuffleSplit(test_size=cfg.TEST_SPLIT, random_state=cfg.SEED)
    tr_idx, te_idx = next(gss.split(df, groups=df["Patient ID"]))
    train_full = df.iloc[tr_idx].reset_index(drop=True)
    test_df    = df.iloc[te_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(test_size=cfg.VAL_SPLIT, random_state=cfg.SEED)
    tr2_idx, va_idx = next(gss2.split(train_full, groups=train_full["Patient ID"]))
    train_df = train_full.iloc[tr2_idx].reset_index(drop=True)
    val_df   = train_full.iloc[va_idx].reset_index(drop=True)

    return train_df, val_df, test_df

df_all = build_dataframe(cfg)
train_df, val_df, test_df = group_split(df_all, cfg)

print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
print("Train pos:", train_df[cfg.TARGETS].sum().astype(int).to_dict(), "NoFinding:", int(train_df["is_nofinding"].sum()))
print("Val pos  :", val_df[cfg.TARGETS].sum().astype(int).to_dict(), "NoFinding:", int(val_df["is_nofinding"].sum()))

# ============================================================
# Imbalance handling: per-class pos_weight
# ============================================================
def compute_pos_weights(df: pd.DataFrame, targets):
    y = df[targets].values.astype(np.float32)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pos_weight = (neg / (pos + 1e-6)).astype(np.float32)
    return pos_weight

POS_W = compute_pos_weights(train_df, cfg.TARGETS)
print("[INFO] pos_weight:", dict(zip(cfg.TARGETS, POS_W.tolist())))

# ============================================================
# tf.data pipeline
# ============================================================
AUTOTUNE = tf.data.AUTOTUNE

def decode_png(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, cfg.IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img

def augment(img):
    # Mild augmentation for CXR
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.05)
    img = tf.image.random_contrast(img, 0.95, 1.05)
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.01)
    img = img + noise
    return tf.clip_by_value(img, 0.0, 1.0)

def make_ds(df: pd.DataFrame, training=True):
    X = df["path"].astype(str).values
    Y = df[cfg.TARGETS].values.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if training:
        ds = ds.shuffle(min(len(df), 8192), seed=cfg.SEED, reshuffle_each_iteration=True)

    def _map(p, y):
        img = decode_png(p)
        if training:
            img = augment(img)
        return img, y

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(cfg.BATCH).prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(train_df, True)
val_ds   = make_ds(val_df, False)
test_ds  = make_ds(test_df, False)

# ============================================================
# Model building
# ============================================================
def build_model(name: str):
    name = name.lower()

    if name == "mobilenet":
        Base = tf.keras.applications.MobileNetV2
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif name == "resnet":
        Base = tf.keras.applications.ResNet50
        preprocess = tf.keras.applications.resnet.preprocess_input
    elif name == "densenet":
        Base = tf.keras.applications.DenseNet121
        preprocess = tf.keras.applications.densenet.preprocess_input
    else:
        raise ValueError("name must be mobilenet/resnet/densenet")

    inp = tf.keras.Input(shape=(*cfg.IMG_SIZE, 3), name="image")
    x = tf.keras.layers.Rescaling(255.0)(inp)
    x = tf.keras.layers.Activation(preprocess)(x)

    backbone = Base(include_top=False, weights="imagenet", input_tensor=x)
    backbone._name = "backbone"
    backbone.trainable = False

    x = backbone.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    # logits (no sigmoid)
    logits = tf.keras.layers.Dense(len(cfg.TARGETS), activation=None, name="logits")(x)

    model = tf.keras.Model(inp, logits, name=f"{name}_multilabel_logits")
    return model

def unfreeze_last_pct(model, pct=0.2, keep_bn_frozen=True):
    """
    Unfreeze last pct layers of the backbone.
    """
    backbone = None
    for lyr in model.layers:
        if lyr.name == "backbone" and isinstance(lyr, tf.keras.Model):
            backbone = lyr
            break

    if backbone is None:
        for lyr in reversed(model.layers):
            if isinstance(lyr, tf.keras.Model):
                backbone = lyr
                break

    target = backbone if backbone is not None else model
    target.trainable = True

    layers = target.layers
    n = len(layers)
    k = max(1, int(n * float(pct)))
    cut = n - k

    for i, lyr in enumerate(layers):
        lyr.trainable = (i >= cut)

    if keep_bn_frozen:
        for lyr in layers:
            if isinstance(lyr, tf.keras.layers.BatchNormalization):
                lyr.trainable = False

    print(f"[unfreeze] target={target.name} total_layers={n} unfrozen_last={k} pct={pct}")

# ============================================================
# Loss: Weighted Focal Loss (from logits) + pos_weight
# ============================================================
class WeightedFocalLoss(tf.keras.losses.Loss):
    def __init__(self, pos_weight, gamma=2.0, name="weighted_focal"):
        super().__init__(name=name)
        self.pos_weight = tf.constant(pos_weight, dtype=tf.float32)
        self.gamma = float(gamma)

    def call(self, y_true, logits):
        y_true = tf.cast(y_true, tf.float32)

        # weighted BCE
        bce = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=logits,
            pos_weight=self.pos_weight
        )  # (B,C)

        # focal modulation
        p = tf.sigmoid(logits)
        p_t = y_true * p + (1.0 - y_true) * (1.0 - p)
        focal = tf.pow(1.0 - p_t, self.gamma)

        loss = focal * bce
        return tf.reduce_mean(loss)

# ============================================================
# Compile helpers
# ============================================================
def compile_model(model, lr, pos_weight, stage="head"):
    if stage == "head":
        opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)

    loss_fn = WeightedFocalLoss(pos_weight=pos_weight, gamma=cfg.FOCAL_GAMMA)

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.AUC(curve="PR", multi_label=True, num_labels=len(cfg.TARGETS), name="auc_pr"),
            tf.keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=len(cfg.TARGETS), name="auc_roc"),
        ],
    )

# ============================================================
# Training
# ============================================================
def train_one(name: str):
    model = build_model(name)

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc_pr", mode="max", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc_pr", mode="max", factor=0.5, patience=1, min_lr=1e-6),
    ]

    # Stage 1: head
    compile_model(model, cfg.LR1, POS_W, stage="head")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=cfg.E_HEAD, callbacks=cbs, verbose=1)

    # Stage 2: fine-tune last pct
    unfreeze_last_pct(model, pct=cfg.UNFREEZE_PCT, keep_bn_frozen=cfg.KEEP_BN_FROZEN)
    compile_model(model, cfg.LR2, POS_W, stage="fine")
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=cfg.E_FINE, callbacks=cbs, verbose=1)

    hist = {k: h1.history.get(k, []) + h2.history.get(k, []) for k in set(h1.history) | set(h2.history)}
    model.save(cfg.RUNS_DIR / f"{name}_final.keras")
    return model, hist

# ============================================================
# Evaluation helpers
# ============================================================
def collect_preds(model, ds):
    y_true, y_prob = [], []
    for xb, yb in ds:
        logits = model.predict(xb, verbose=0)
        prob = tf.sigmoid(logits).numpy()
        y_true.append(yb.numpy())
        y_prob.append(prob)
    return np.vstack(y_true), np.vstack(y_prob)

def tune_thresholds(y_true, y_prob):
    # per-class threshold maximizing F2 on validation (favor recall)
    thr = []
    for k in range(y_true.shape[1]):
        best_t, best_f2 = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 19):
            y_hat = (y_prob[:, k] >= t).astype(int)
            tp = ((y_hat == 1) & (y_true[:, k] == 1)).sum()
            fp = ((y_hat == 1) & (y_true[:, k] == 0)).sum()
            fn = ((y_hat == 0) & (y_true[:, k] == 1)).sum()
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f2 = (5 * prec * rec) / (4 * prec + rec + 1e-9)
            if f2 > best_f2:
                best_f2 = f2
                best_t = float(t)
        thr.append(best_t)
    return np.array(thr, dtype=np.float32)

def per_class_confusion(y_true_bin, y_pred_bin, targets):
    mats = {}
    for i, t in enumerate(targets):
        yt = y_true_bin[:, i].astype(int)
        yp = y_pred_bin[:, i].astype(int)
        tp = int(((yp == 1) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        mats[t] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
    return mats

def plot_confusion_heatmaps(mats, targets, out_png):
    n = len(targets)
    plt.figure(figsize=(12, 2.5 * n))
    for i, t in enumerate(targets, start=1):
        tn, fp, fn, tp = mats[t]["TN"], mats[t]["FP"], mats[t]["FN"], mats[t]["TP"]
        M = np.array([[tn, fp], [fn, tp]])
        plt.subplot(n, 1, i)
        plt.imshow(M)
        plt.title(f"{t}  [[TN,FP],[FN,TP]]")
        for r in range(2):
            for c in range(2):
                plt.text(c, r, str(M[r, c]), ha="center", va="center")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ============================================================
# Main: Train all models + plot compare + eval
# ============================================================
def main():
    hist_all = {}
    models = {}

    for name in ["mobilenet", "resnet", "densenet"]:
        print("\n=== TRAIN:", name, "===")
        m, hist = train_one(name)
        models[name] = m
        hist_all[name] = hist

    # Plot compare validation PR-AUC
    plt.figure(figsize=(10, 4))
    for name, h in hist_all.items():
        if "val_auc_pr" in h:
            plt.plot(h["val_auc_pr"], label=f"{name} val_auc_pr")
    plt.title("Compare Validation PR-AUC")
    plt.xlabel("epoch")
    plt.ylabel("val_auc_pr")
    plt.legend()
    plt.tight_layout()
    out_plot = cfg.RUNS_DIR / "compare_val_auc_pr.png"
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print("[DONE] Saved plot:", out_plot)

    results = {}

    for name, model in models.items():
        print("\n=== EVAL:", name, "===")

        # tune thresholds on val
        yv_true, yv_prob = collect_preds(model, val_ds)
        thr = tune_thresholds(yv_true, yv_prob)

        # test preds
        yt_true, yt_prob = collect_preds(model, test_ds)
        yt_pred = (yt_prob >= thr[None, :]).astype(int)

        roc = roc_auc_score(yt_true, yt_prob, average="macro")
        pr  = average_precision_score(yt_true, yt_prob, average="macro")

        report = classification_report(
            yt_true.astype(int),
            yt_pred.astype(int),
            target_names=cfg.TARGETS,
            digits=4,
            zero_division=0
        )

        mats = per_class_confusion(yt_true, yt_pred, cfg.TARGETS)

        # save report
        report_path = cfg.RUNS_DIR / f"{name}_report.txt"
        report_path.write_text(
            f"MODEL: {name}\n"
            f"ROC_AUC_macro: {roc:.4f}\n"
            f"PR_AUC_macro: {pr:.4f}\n"
            f"Thresholds: {dict(zip(cfg.TARGETS, thr.tolist()))}\n\n"
            f"{report}\n\n"
            f"Per-class confusion (TP,TN,FP,FN):\n{mats}\n",
            encoding="utf-8"
        )
        print("[INFO] Saved:", report_path)

        # save confusions plot
        conf_png = cfg.RUNS_DIR / f"{name}_confusions.png"
        plot_confusion_heatmaps(mats, cfg.TARGETS, conf_png)
        print("[INFO] Saved:", conf_png)

        results[name] = {"roc_auc_macro": roc, "pr_auc_macro": pr, "thresholds": thr, "conf": mats}

    # choose best model by PR-AUC
    best_name = max(results.keys(), key=lambda k: results[k]["pr_auc_macro"])
    best_path = cfg.RUNS_DIR / "best_by_pr_auc.txt"
    best_path.write_text(
        f"Best model by PR_AUC_macro: {best_name}\n"
        f"PR_AUC_macro: {results[best_name]['pr_auc_macro']:.4f}\n"
        f"ROC_AUC_macro: {results[best_name]['roc_auc_macro']:.4f}\n"
        f"Thresholds: {dict(zip(cfg.TARGETS, results[best_name]['thresholds'].tolist()))}\n",
        encoding="utf-8"
    )
    print("[DONE] Best by PR-AUC:", best_name)
    print("[DONE] Saved:", best_path)
    print("[DONE] All outputs in:", cfg.RUNS_DIR)

if __name__ == "__main__":
    main()
