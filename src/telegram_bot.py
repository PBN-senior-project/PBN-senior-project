import os, time, requests
from datetime import datetime

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PROM = "http://prometheus:9090"
BASE = f"https://api.telegram.org/bot{TOKEN}"

def query(metric):
    try:
        x = requests.get(f"{PROM}/api/v1/query",
            params={"query": metric}, timeout=10).json()["data"]["result"]
        return (float(x[0]["value"][1]), x[0].get("metric", {})) if x else (None, {})
    except Exception:
        return None, {}

def send(chat, text):
    try:
        requests.post(f"{BASE}/sendMessage",
            json={"chat_id": chat, "text": text}, timeout=10)
    except Exception as e:
        print("Telegram error:", e)

def duration(s):
    if s is None: return "-"
    s = max(0, int(s))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

def date(ts):
    return datetime.fromtimestamp(ts).strftime("%d %b %Y %H:%M") if ts else "-"

def get_values():
    names = [
        "training_status", "training_epoch", "training_total_epochs",
        "training_batch", "training_total_batches",
        "training_images_processed", "training_total_images",
        "training_progress_percent", "training_loss", "training_auc",
        "validation_loss", "validation_auc",
        "training_seconds_per_step", "training_epoch_elapsed_seconds",
        "training_train_remaining_seconds",
        "training_validation_estimate_seconds",
        "training_next_epoch_eta_timestamp",
        "training_model_elapsed_seconds",
        "training_model_remaining_seconds",
        "training_model_finish_timestamp",
        "training_last_update_timestamp"
    ]
    v = {n: query(n)[0] for n in names}
    _, labels = query("training_epoch")
    return v, labels.get("model", "Unknown")

def health(v):
    epoch, va = v["training_epoch"], v["validation_auc"]
    if epoch and epoch <= 2:
        return "🟡 Model Health: EARLY TRAINING\nยังเร็วเกินไปที่จะสรุป"
    if va is None:
        return "🟡 Model Health: MONITORING"
    if va < .55: return "🔴 Model Health: POOR"
    if va < .70: return "🟠 Model Health: WARNING"
    if va >= .80: return "🟢 Model Health: GOOD"
    return "🟡 Model Health: ACCEPTABLE"

def get_status():
    v, model = get_values()
    if v["training_epoch"] is None:
        return "⚪ ยังไม่พบ Training Metrics"

    status = {1: "🟢 TRAINING", 2: "🔵 VALIDATING", 3: "✅ COMPLETED"}
    return (
        "🧠 PBN Chest X-ray Training\n\n"
        f"Status: {status.get(int(v['training_status'] or 0), '⚪ UNKNOWN')}\n"
        f"Model: {model}\n"
        f"Epoch: {int(v['training_epoch'])} / {int(v['training_total_epochs'] or 0)}\n"
        f"Batch: {int(v['training_batch'] or 0):,} / {int(v['training_total_batches'] or 0):,}\n"
        f"Progress: {(v['training_progress_percent'] or 0):.2f}%\n\n"
        f"Loss: {(v['training_loss'] or 0):.4f}\n"
        f"AUC: {(v['training_auc'] or 0):.4f}"
    )

def get_report():
    v, model = get_values()
    if v["training_epoch"] is None:
        return "⚪ ยังไม่พบ Training Metrics"

    e = int(v["training_epoch"])
    total_e = int(v["training_total_epochs"] or 0)
    status = {1: "🟢 TRAINING", 2: "🔵 VALIDATING", 3: "✅ COMPLETED"}

    text = (
        "🧠 PBN Chest X-ray Training\n\n"
        f"Status: {status.get(int(v['training_status'] or 0), '⚪ UNKNOWN')}\n"
        f"Model: {model}\n\n"

        f"Epoch: {e} / {total_e}\n"
        f"Batch: {int(v['training_batch'] or 0):,} / "
        f"{int(v['training_total_batches'] or 0):,}\n"
        f"Images: {int(v['training_images_processed'] or 0):,} / "
        f"{int(v['training_total_images'] or 0):,}\n"
        f"Epoch Progress: {(v['training_progress_percent'] or 0):.2f}%\n\n"

        f"⏱ Epoch elapsed: {duration(v['training_epoch_elapsed_seconds'])}\n"
        f"⏳ Train remaining: ~{duration(v['training_train_remaining_seconds'])}\n"
        f"🔍 Estimated validation: ~{duration(v['training_validation_estimate_seconds'])}\n"
    )

    if e < total_e:
        text += (
            f"➡️ Epoch {e + 1} expected start:\n"
            f"   ~{date(v['training_next_epoch_eta_timestamp'])}\n"
        )

    text += (
        f"\n⏱ Model elapsed: {duration(v['training_model_elapsed_seconds'])}\n"
        f"⏳ Model remaining: ~{duration(v['training_model_remaining_seconds'])}\n"
        f"🏁 {model} expected finish:\n"
        f"   ~{date(v['training_model_finish_timestamp'])}\n\n"

        f"⚡ Speed: {(v['training_seconds_per_step'] or 0):.2f} sec/step\n\n"

        "📊 Metrics\n"
        f"Loss: {(v['training_loss'] or 0):.4f}\n"
        f"Train AUC: {(v['training_auc'] or 0):.4f}\n"
    )

    if v["validation_loss"] is not None:
        text += f"Val Loss: {v['validation_loss']:.4f}\n"
    if v["validation_auc"] is not None:
        text += f"Val AUC: {v['validation_auc']:.4f}\n"

    text += (
        f"\n{health(v)}\n\n"
        f"Last update: "
        f"{duration(time.time() - v['training_last_update_timestamp'])} ago"
        if v["training_last_update_timestamp"] else ""
    )

    return text

def handle(msg):
    chat, cmd = msg["chat"]["id"], msg.get("text", "").strip().split("@")[0]

    if cmd == "/status": send(chat, get_status())
    elif cmd in ("/report", "/metrics"): send(chat, get_report())
    elif cmd == "/help":
        send(chat,
            "📋 Commands\n\n"
            "/status - สถานะแบบย่อ\n"
            "/report - รายงาน Training แบบละเอียด\n"
            "/metrics - รายงานแบบละเอียด\n"
            "/help - ดูคำสั่ง")
    else:
        send(chat, "ใช้ /status หรือ /report")

def main():
    print("🤖 Telegram monitoring bot started")
    offset = None

    while True:
        try:
            p = {"timeout": 30}
            if offset is not None: p["offset"] = offset

            updates = requests.get(
                f"{BASE}/getUpdates", params=p, timeout=35
            ).json().get("result", [])

            for u in updates:
                offset = u["update_id"] + 1
                if "message" in u: handle(u["message"])
        except Exception as e:
            print("Telegram error:", e)
            time.sleep(5)

if __name__ == "__main__":
    main()