import os, glob, time, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, ResNet50V2, MobileNetV2
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

# ========================= CONFIG =========================
mixed_precision.set_global_policy("float32")
CSV_PATH, ARCHIVE_DIR = "/archive/Data_Entry_2017.csv", "/archive"
MODEL_DIR, GRAPH_DIR = "/app/models_v7", "/app/outputs/graphs_v7"
IMG_SIZE, BATCH_SIZE, EPOCHS, LR = (384, 384), 8, 10, 1e-4
N_NO_FINDING, N_PER_CLASS_VIEW, SEED = 5000, 1895, 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.10, 0.20
CLASSES = ["Infiltration", "Effusion", "Atelectasis", "Nodule", "Mass", "Pneumothorax"]
VIEWS, MODEL_NAMES = ["PA", "AP"], ["DenseNet121", "ResNet50V2", "MobileNetV2"]
BACKBONES = {"DenseNet121": DenseNet121, "ResNet50V2": ResNet50V2, "MobileNetV2": MobileNetV2}
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1) < 1e-8
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(GRAPH_DIR, exist_ok=True)

# ========================= DATA =========================
def load_data():
    print("⏳ Loading Data...")
    df = pd.read_csv(CSV_PATH)
    paths = glob.glob(os.path.join(ARCHIVE_DIR, "**", "*.png"), recursive=True)
    df["filepath"] = df["Image Index"].map({os.path.basename(p): p for p in paths})
    print(f"📁 Images: {len(paths)} | ✅ Matched: {df.filepath.notna().sum()} | ❌ Missing: {df.filepath.isna().sum()}")
    df = df[df.filepath.notna()].copy()
    df["View Position"] = df["View Position"].astype(str).str.strip()
    df = df[df["View Position"].isin(VIEWS)].copy()
    labels = df["Finding Labels"].fillna("").str.split("|")
    for c in CLASSES: df[c] = labels.apply(lambda x, c=c: int(c in x))
    print(df["View Position"].value_counts())
    return df

def balance_positive(df):
    pos = df[df[CLASSES].sum(axis=1) > 0].drop_duplicates("Image Index").reset_index(drop=True)
    names, rows = [], []
    for c in CLASSES:
        for v in VIEWS:
            names.append(f"{c}_{v}")
            rows.append(((pos[c].values == 1) & (pos["View Position"].values == v)).astype(float))
    A = csr_matrix(np.vstack(rows)); target = np.full(len(names), N_PER_CLASS_VIEW, float)
    result = milp(
        c=np.zeros(len(pos)), integrality=np.ones(len(pos), int),
        bounds=Bounds(np.zeros(len(pos)), np.ones(len(pos))),
        constraints=LinearConstraint(A, target, target), options={"time_limit": 300}
    )
    if not result.success:
        raise RuntimeError(f"❌ Exact balance failed: {result.message}")
    out = pos[result.x > .5].copy().reset_index(drop=True)
    for c in CLASSES:
        counts = out[out[c] == 1]["View Position"].value_counts()
        pa, ap = int(counts.get("PA", 0)), int(counts.get("AP", 0))
        print(f"{c:15} PA={pa:4} AP={ap:4} Total={pa+ap:4}")
        if pa != N_PER_CLASS_VIEW or ap != N_PER_CLASS_VIEW:
            raise RuntimeError("❌ Balance verification failed")
    return out

def make_dataset(df):
    positive = balance_positive(df)
    negative = (df[df["Finding Labels"].eq("No Finding")]
                .drop_duplicates("Image Index")
                .sample(n=min(N_NO_FINDING, (df["Finding Labels"] == "No Finding").sum()), random_state=SEED))
    selected = (pd.concat([positive, negative], ignore_index=True)
                .drop_duplicates("Image Index")
                .sample(frac=1, random_state=SEED).reset_index(drop=True))
    print(f"📦 Selected: {len(selected)} | 👤 Patients: {selected['Patient ID'].nunique()}")
    return selected

# ========================= PATIENT SPLIT =========================
def patient_split(df):
    rows = []
    for pid, g in df.groupby("Patient ID"):
        r = {"Patient ID": pid}
        for c in CLASSES:
            for v in VIEWS:
                r[f"{c}_{v}"] = int(((g[c] == 1) & (g["View Position"] == v)).any())
        rows.append(r)
    pdf = pd.DataFrame(rows)
    strat = [f"{c}_{v}" for c in CLASSES for v in VIEWS]
    X, Y = pdf[["Patient ID"]].values, pdf[strat].values

    s1 = MultilabelStratifiedShuffleSplit(1, test_size=VAL_RATIO + TEST_RATIO, random_state=SEED)
    train_i, hold_i = next(s1.split(X, Y))
    hold = pdf.iloc[hold_i].reset_index(drop=True)
    hold_y = hold[strat].values
    val_frac = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    s2 = MultilabelStratifiedShuffleSplit(1, test_size=1 - val_frac, random_state=SEED)
    val_i, test_i = next(s2.split(hold[["Patient ID"]].values, hold_y))

    train_ids = set(pdf.iloc[train_i]["Patient ID"])
    val_ids = set(hold.iloc[val_i]["Patient ID"])
    test_ids = set(hold.iloc[test_i]["Patient ID"])

    assert train_ids.isdisjoint(val_ids), "❌ Train/Validation patient leakage"
    assert train_ids.isdisjoint(test_ids), "❌ Train/Test patient leakage"
    assert val_ids.isdisjoint(test_ids), "❌ Validation/Test patient leakage"
    print("✅ No patient leakage")

    train = df[df["Patient ID"].isin(train_ids)].copy()
    val = df[df["Patient ID"].isin(val_ids)].copy()
    test = df[df["Patient ID"].isin(test_ids)].copy()
    print(f"Train={len(train)} | Val={len(val)} | Test={len(test)}")
    print(f"Patients: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    return train, val, test

# ========================= PREPROCESSING =========================
# No CLAHE / Brightness / Gamma augmentation
def train_preprocess(img):
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Random crop 384 -> 368 -> 384
    h, w = img.shape[:2]
    ch, cw = min(368, h), min(368, w)
    top = np.random.randint(0, h - ch + 1) if h > ch else 0
    left = np.random.randint(0, w - cw + 1) if w > cw else 0

    img = img[top:top + ch, left:left + cw]
    img = cv2.resize(img, IMG_SIZE).astype(np.float32) / 255.0

    return np.clip(img, 0, 1)


def eval_preprocess(img):
    # Validation/Test: normalization only
    return np.clip(img, 0, 255).astype(np.float32) / 255.0


def make_generators(train, val, test):
    tr_aug = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        shear_range=5,
        horizontal_flip=True,
        fill_mode="constant",
        cval=0,
        preprocessing_function=train_preprocess
    )
    ev_aug = ImageDataGenerator(
        preprocessing_function=eval_preprocess
    )

    common = dict(
        x_col="filepath",
        y_col=CLASSES,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="raw"
    )

    tr = tr_aug.flow_from_dataframe(
        train, shuffle=True, seed=SEED, **common
    )
    va = ev_aug.flow_from_dataframe(
        val, shuffle=False, **common
    )
    te = ev_aug.flow_from_dataframe(
        test, shuffle=False, **common
    )
    return tr, va, te

# ========================= LOSS + MODEL =========================
def weighted_focal_loss(pos_weights, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1 - 1e-7)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha = y_true * pos_weights + (1 - y_true)
        return tf.reduce_mean(-alpha * tf.pow(1 - pt, gamma) * tf.math.log(pt))
    return loss

def build_model(name, pos_weights):
    base = BACKBONES[name](include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    base.trainable = True
    for layer in base.layers[:-120]: layer.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x); x = layers.Dropout(.5)(x)
    out = layers.Dense(len(CLASSES), activation="sigmoid", dtype="float32")(x)
    model = models.Model(base.input, out, name=name)
    model.compile(optimizers.AdamW(LR, weight_decay=1e-4), weighted_focal_loss(pos_weights),
                  metrics=[tf.keras.metrics.AUC(multi_label=True, name="auc")])
    return model

# ========================= PROMETHEUS =========================
class PrometheusTrainingCallback(callbacks.Callback):
    NAMES = ["training_status","training_epoch","training_total_epochs","training_batch","training_total_batches",
             "training_images_processed","training_total_images","training_progress_percent","training_loss","training_auc",
             "validation_loss","validation_auc","training_seconds_per_step","training_epoch_elapsed_seconds",
             "training_train_remaining_seconds","training_validation_estimate_seconds","training_next_epoch_eta_timestamp",
             "training_model_elapsed_seconds","training_model_remaining_seconds","training_model_finish_timestamp",
             "training_last_update_timestamp"]
    def __init__(self, name, total_images, total_epochs, total_batches, val_batches):
        super().__init__(); self.name=name; self.total_images=total_images; self.total_epochs=total_epochs
        self.total_batches=total_batches; self.val_batches=val_batches; self.val_times=[]
        self.registry=CollectorRegistry(); self.m={n:Gauge(n,n,["model"],registry=self.registry) for n in self.NAMES}
    def set(self,n,v): self.m[n].labels(model=self.name).set(float(v))
    def push(self):
        try: push_to_gateway("pushgateway:9091", job="chestxray_training", registry=self.registry)
        except Exception as e: print("⚠️ Prometheus:", e)
    def sets(self,d):
        for k,v in d.items(): self.set(k,v)
    def on_train_begin(self, logs=None):
        self.model_start=time.time(); self.sets({"training_status":1,"training_total_epochs":self.total_epochs,
            "training_total_batches":self.total_batches,"training_total_images":self.total_images,
            "training_last_update_timestamp":self.model_start}); self.push()
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch=epoch+1; self.epoch_start=time.time(); self.sets({"training_status":1,"training_epoch":self.epoch,
            "training_batch":0,"training_images_processed":0,"training_progress_percent":0,
            "training_last_update_timestamp":self.epoch_start}); self.push()
    def on_train_batch_end(self, batch, logs=None):
        logs=logs or {}; now=time.time(); b=batch+1; elapsed=now-self.epoch_start; sec=elapsed/b
        rem=max(self.total_batches-b,0)*sec; val_est=np.mean(self.val_times) if self.val_times else self.val_batches*sec
        epoch_est=self.total_batches*sec+val_est; model_rem=rem+val_est+max(self.total_epochs-self.epoch,0)*epoch_est
        d={"training_batch":b,"training_images_processed":min(b*BATCH_SIZE,self.total_images),
           "training_progress_percent":b/self.total_batches*100,"training_seconds_per_step":sec,
           "training_epoch_elapsed_seconds":elapsed,"training_train_remaining_seconds":rem,
           "training_validation_estimate_seconds":val_est,"training_next_epoch_eta_timestamp":now+rem+val_est,
           "training_model_elapsed_seconds":now-self.model_start,"training_model_remaining_seconds":model_rem,
           "training_model_finish_timestamp":now+model_rem,"training_last_update_timestamp":now}
        if "loss" in logs: d["training_loss"]=logs["loss"]
        if "auc" in logs: d["training_auc"]=logs["auc"]
        self.sets(d)
        if b==1 or b%50==0 or b==self.total_batches: self.push()
    def on_test_begin(self, logs=None):
        self.val_start=time.time(); self.sets({"training_status":2,"training_last_update_timestamp":self.val_start}); self.push()
    def on_test_end(self, logs=None):
        now=time.time(); self.val_times.append(now-self.val_start)
        self.sets({"training_validation_estimate_seconds":np.mean(self.val_times),"training_status":1,
                   "training_last_update_timestamp":now}); self.push()
    def on_epoch_end(self, epoch, logs=None):
        mp={"loss":"training_loss","auc":"training_auc","val_loss":"validation_loss","val_auc":"validation_auc"}
        self.sets({v:(logs or {})[k] for k,v in mp.items() if k in (logs or {})})
        self.set("training_last_update_timestamp",time.time()); self.push()
    def on_train_end(self, logs=None):
        now=time.time(); self.sets({"training_status":3,"training_model_remaining_seconds":0,
            "training_model_finish_timestamp":now,"training_last_update_timestamp":now}); self.push()

# ========================= TRAIN =========================
def train_models(train_gen, val_gen, pos_weights):
    paths, histories = [], {}
    for name in MODEL_NAMES:
        print(f"\n🧠 Training {name}"); model=build_model(name, pos_weights)
        path=os.path.join(MODEL_DIR,f"best_{name}_v7.keras"); paths.append(path)
        cb=[callbacks.EarlyStopping("val_auc",mode="max",patience=5,restore_best_weights=True),
            callbacks.ReduceLROnPlateau("val_auc",mode="max",factor=.1,patience=2,min_lr=1e-6,verbose=1),
            callbacks.ModelCheckpoint(path,monitor="val_auc",mode="max",save_best_only=True,verbose=1),
            PrometheusTrainingCallback(name,len(train_gen.dataframe),EPOCHS,len(train_gen),len(val_gen))]
        h=model.fit(train_gen,validation_data=val_gen,epochs=EPOCHS,callbacks=cb)
        histories[name]=h.history["val_auc"]
        del model; tf.keras.backend.clear_session()
    return paths, histories

# ========================= EVALUATION =========================
def save_training_graph(histories):
    plt.figure(figsize=(10,6))
    for n,v in histories.items(): plt.plot(v,marker="o",label=n)
    plt.title("V7 Validation AUC Comparison (6 Findings)"); plt.xlabel("Epoch"); plt.ylabel("Validation AUC")
    plt.legend(); plt.grid(alpha=.3); plt.savefig(os.path.join(GRAPH_DIR,"training_comparison_v7.png"),bbox_inches="tight",dpi=300); plt.close()

def ensemble_predict(paths, gen):
    pred = np.zeros((gen.samples, len(CLASSES)), dtype=np.float32)
    gen.reset()

    for p in paths:
        print("⏳ Predicting:", os.path.basename(p))
        model = tf.keras.models.load_model(p, compile=False)
        gen.reset()
        pred += model.predict(gen, verbose=1)[:gen.samples]
        del model
        tf.keras.backend.clear_session()

    return pred / len(paths)


def find_thresholds(y_true, pred):
    """Choose one threshold per disease using Validation F1 only."""
    thresholds = []
    print("\n🎯 Thresholds selected from VALIDATION")

    for i, disease in enumerate(CLASSES):
        best_t, best_f1 = 0.5, -1.0

        for t in np.arange(0.05, 0.96, 0.01):
            y_hat = (pred[:, i] >= t).astype(int)
            score = f1_score(
                y_true[:, i],
                y_hat,
                zero_division=0
            )

            if score > best_f1:
                best_f1 = score
                best_t = float(t)

        thresholds.append(best_t)
        print(
            f"{disease:15} "
            f"Threshold={best_t:.2f} "
            f"Validation F1={best_f1:.4f}"
        )

    return np.asarray(thresholds, dtype=np.float32)


def evaluate(y_true, pred, thresholds, split_name="TEST"):
    lines = [f"V7 ENSEMBLE {split_name} AUC SCORES", "-" * 40]
    aucs = []

    for i, c in enumerate(CLASSES):
        try:
            auc = roc_auc_score(y_true[:, i], pred[:, i])
            aucs.append(auc)
            lines.append(f"{c:20}: {auc:.4f}")
        except ValueError:
            lines.append(f"{c:20}: N/A")

    macro_auc = np.mean(aucs) if aucs else float("nan")
    lines.append(f"\nMACRO AUC: {macro_auc:.4f}")
    auc_text = "\n".join(lines)
    print("\n" + auc_text)

    with open(
        os.path.join(GRAPH_DIR, f"V7_{split_name}_AUC_Scores.txt"),
        "w"
    ) as f:
        f.write(auc_text)

    # Thresholds are selected on Validation, then locked for Test.
    y_pred = (pred >= thresholds).astype(int)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        zero_division=0
    )

    with open(
        os.path.join(GRAPH_DIR, f"V7_{split_name}_Classification_Report.txt"),
        "w"
    ) as f:
        f.write("Thresholds optimized on Validation Set\n")
        for disease, t in zip(CLASSES, thresholds):
            f.write(f"{disease}: {t:.2f}\n")
        f.write("\n")
        f.write(report)

    cm_dir = os.path.join(
        GRAPH_DIR,
        f"Confusion_Matrices_{split_name}"
    )
    os.makedirs(cm_dir, exist_ok=True)

    for i, c in enumerate(CLASSES):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(
            f"{c:15} "
            f"Sensitivity={sensitivity:.4f} "
            f"Specificity={specificity:.4f}"
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm)

        for r in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(
                    col, r, cm[r, col],
                    ha="center", va="center"
                )

        ax.set(
            title=f"Confusion Matrix: {c}",
            ylabel="Actual",
            xlabel="Predicted",
            xticks=[0, 1],
            yticks=[0, 1]
        )
        fig.colorbar(im, ax=ax)

        fig.savefig(
            os.path.join(cm_dir, f"CM_{c}.png"),
            bbox_inches="tight",
            dpi=150
        )
        plt.close(fig)

# ========================= MAIN =========================
def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # 1) Load + exact balance
    selected = make_dataset(load_data())

    # 2) Patient-level split
    train, val, test = patient_split(selected)

    # 3) Positive weights from Train only
    pos = train[CLASSES].sum().values
    pos_weights = tf.constant(
        len(train) / (2.0 * (pos + 1e-5)),
        dtype=tf.float32
    )

    # 4) Generators
    train_gen, val_gen, test_gen = make_generators(
        train, val, test
    )

    # 5) Train and select best model using Validation AUC
    paths, histories = train_models(
        train_gen,
        val_gen,
        pos_weights
    )
    save_training_graph(histories)

    # 6) Validation -> optimize threshold for each disease
    val_pred = ensemble_predict(paths, val_gen)
    val_true = val_gen.labels[:val_gen.samples]
    thresholds = find_thresholds(val_true, val_pred)

    threshold_path = os.path.join(
        GRAPH_DIR,
        "V7_Validation_Thresholds.txt"
    )
    with open(threshold_path, "w") as f:
        for disease, t in zip(CLASSES, thresholds):
            f.write(f"{disease}: {t:.2f}\n")

    # 7) Final Test using locked Validation thresholds
    test_pred = ensemble_predict(paths, test_gen)
    test_true = test_gen.labels[:test_gen.samples]

    evaluate(
        test_true,
        test_pred,
        thresholds,
        split_name="TEST"
    )

    print(f"\n✅ Thresholds saved: {threshold_path}")
    print(f"✅ Results saved: {GRAPH_DIR}")


if __name__ == "__main__":
    main()
