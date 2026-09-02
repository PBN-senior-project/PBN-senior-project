import os
import time
import threading
import requests
from datetime import datetime

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PROM = "http://prometheus:9090"
BASE = f"https://api.telegram.org/bot{TOKEN}"

# ถ้ามีหลายคน ให้ใส่ TELEGRAM_CHAT_IDS=123456,789012
ENV_CHAT_IDS = os.getenv("TELEGRAM_CHAT_IDS", "")

SUBSCRIBERS = {
    int(x.strip())
    for x in ENV_CHAT_IDS.split(",")
    if x.strip()
}

CHECK_INTERVAL = 10

last_epoch = None
last_model = None


# =========================================================
# PROMETHEUS
# =========================================================
def query(metric):
    try:
        r = requests.get(
            f"{PROM}/api/v1/query",
            params={"query": metric},
            timeout=10
        )
        r.raise_for_status()

        x = r.json()["data"]["result"]

        if not x:
            return None, {}

        return float(x[0]["value"][1]), x[0].get("metric", {})

    except Exception as e:
        print("Prometheus error:", e)
        return None, {}


# =========================================================
# TELEGRAM
# =========================================================
def send(chat, text):
    try:
        r = requests.post(
            f"{BASE}/sendMessage",
            json={
                "chat_id": chat,
                "text": text
            },
            timeout=10
        )
        r.raise_for_status()

    except Exception as e:
        print("Telegram send error:", e)


def broadcast(text):
    """ส่งข้อความให้ทุกคนที่ subscribe"""
    for chat in list(SUBSCRIBERS):
        send(chat, text)


# =========================================================
# GET TRAINING VALUES
# =========================================================
def get_values():

    names = [
        "training_status",
        "training_epoch",
        "training_total_epochs",
        "training_batch",
        "training_total_batches",
        "training_progress_percent",
        "training_loss",
        "training_auc",
    ]

    values = {}

    for name in names:
        values[name] = query(name)[0]

    # ดึง model จาก label ของ training_epoch
    _, labels = query("training_epoch")

    model = labels.get("model", "Unknown")

    return values, model


# =========================================================
# FORMAT VALUE
# =========================================================
def value_or_dash(value, decimals=4):

    if value is None:
        return "-"

    return f"{value:.{decimals}f}"


# =========================================================
# STATUS
# =========================================================
def get_status():

    v, model = get_values()

    if v["training_epoch"] is None:
        return "⚪ ยังไม่พบ Training Metrics"

    status_map = {
        1: "🟢 TRAINING",
        2: "🔵 VALIDATING",
        3: "✅ COMPLETED"
    }

    status = status_map.get(
        int(v["training_status"] or 0),
        "⚪ UNKNOWN"
    )

    epoch = int(v["training_epoch"])
    total_epochs = int(v["training_total_epochs"] or 0)

    batch = int(v["training_batch"] or 0)
    total_batches = int(v["training_total_batches"] or 0)

    progress = v["training_progress_percent"]

    return (
        "🧠 PLOY Chest X-ray Training\n\n"

        f"Status: {status}\n"
        f"Model: {model}\n"
        f"Epoch: {epoch} / {total_epochs}\n"
        f"Batch: {batch:,} / {total_batches:,}\n"
        f"Progress: {value_or_dash(progress, 2)}%\n\n"

        f"Loss: {value_or_dash(v['training_loss'])}\n"
        f"AUC: {value_or_dash(v['training_auc'])}"
    )


# =========================================================
# NEW MODEL NOTIFICATION
# =========================================================
def model_notification(model, epoch, total_epochs):

    return (
        "🚀 NEW MODEL STARTED\n\n"

        "🧠 PLOY Chest X-ray Training\n\n"

        f"Model: {model}\n"
        f"Epoch: {epoch} / {total_epochs}\n\n"

        "✅ เริ่ม Training โมเดลใหม่แล้ว"
    )


# =========================================================
# NEW EPOCH NOTIFICATION
# =========================================================
def epoch_notification(model, epoch, total_epochs):

    return (
        "🔔 NEW EPOCH\n\n"

        "🧠 PLOY Chest X-ray Training\n\n"

        f"Model: {model}\n"
        f"Epoch: {epoch} / {total_epochs}\n\n"

        f"▶️ เริ่ม Epoch {epoch} แล้ว"
    )


# =========================================================
# AUTO MONITOR
# =========================================================
def monitor_training():

    global last_epoch, last_model

    print("📡 Training monitor started")

    while True:

        try:

            epoch_value, labels = query("training_epoch")
            total_epochs, _ = query("training_total_epochs")

            if epoch_value is None:
                time.sleep(CHECK_INTERVAL)
                continue

            epoch = int(epoch_value)
            total_epochs = int(total_epochs or 0)

            model = labels.get("model", "Unknown")

            # -----------------------------------------
            # NEW MODEL
            # -----------------------------------------
            if last_model is None:

                last_model = model

                broadcast(
                    model_notification(
                        model,
                        epoch,
                        total_epochs
                    )
                )

            elif model != last_model:

                print(
                    f"🚀 New model: "
                    f"{last_model} -> {model}"
                )

                last_model = model

                broadcast(
                    model_notification(
                        model,
                        epoch,
                        total_epochs
                    )
                )

                # reset epoch เมื่อเปลี่ยน model
                last_epoch = None

            # -----------------------------------------
            # NEW EPOCH
            # -----------------------------------------
            if last_epoch is None:

                last_epoch = epoch

                broadcast(
                    epoch_notification(
                        model,
                        epoch,
                        total_epochs
                    )
                )

            elif epoch != last_epoch:

                print(
                    f"🔔 New epoch: "
                    f"{last_epoch} -> {epoch}"
                )

                last_epoch = epoch

                broadcast(
                    epoch_notification(
                        model,
                        epoch,
                        total_epochs
                    )
                )

        except Exception as e:

            print("Monitor error:", e)

        time.sleep(CHECK_INTERVAL)


# =========================================================
# TELEGRAM COMMAND
# =========================================================
def handle(msg):

    chat = msg["chat"]["id"]

    text = msg.get("text", "").strip()

    cmd = text.split()[0].split("@")[0] if text else ""

    # เมื่อมีคนคุยกับ Bot
    # ให้รับ notification อัตโนมัติ
    SUBSCRIBERS.add(chat)

    if cmd == "/start":

        send(
            chat,
            "✅ PLOY Chest X-ray Monitor Started\n\n"
            "คุณจะได้รับแจ้งเตือนอัตโนมัติเมื่อ:\n"
            "🚀 เริ่ม Model ใหม่\n"
            "🔔 ขึ้น Epoch ใหม่\n\n"
            "ใช้ /status เพื่อดูสถานะ Training"
        )

    elif cmd in ("/status", "/statust"):

        send(
            chat,
            get_status()
        )

    elif cmd == "/help":

        send(
            chat,
            "📋 Commands\n\n"
            "/status - ดูสถานะ Training\n"
            "/help - ดูคำสั่ง"
        )

    else:

        send(
            chat,
            "ใช้ /status เพื่อดูสถานะ Training"
        )


# =========================================================
# TELEGRAM BOT
# =========================================================
def telegram_bot():

    print("🤖 Telegram bot started")

    offset = None

    while True:

        try:

            params = {
                "timeout": 30
            }

            if offset is not None:
                params["offset"] = offset

            response = requests.get(
                f"{BASE}/getUpdates",
                params=params,
                timeout=35
            )

            response.raise_for_status()

            updates = response.json().get(
                "result",
                []
            )

            for update in updates:

                offset = update["update_id"] + 1

                if "message" in update:
                    handle(update["message"])

        except Exception as e:

            print("Telegram error:", e)

            time.sleep(5)


# =========================================================
# MAIN
# =========================================================
def main():

    print("🧠 PLOY Chest X-ray Telegram Monitor")

    # Thread สำหรับตรวจ Epoch / Model
    monitor_thread = threading.Thread(
        target=monitor_training,
        daemon=True
    )

    monitor_thread.start()

    # Telegram command listener
    telegram_bot()


if __name__ == "__main__":
    main()