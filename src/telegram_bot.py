import os
import time
import requests

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PROMETHEUS_URL = "http://prometheus:9090"

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"


def query_prometheus(metric):
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": metric},
            timeout=10
        )
        data = r.json()

        result = data.get("data", {}).get("result", [])

        if not result:
            return None, None

        item = result[0]
        value = float(item["value"][1])
        labels = item.get("metric", {})

        return value, labels

    except Exception:
        return None, None


def send_message(chat_id, text):
    requests.post(
        f"{BASE_URL}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": text
        },
        timeout=10
    )


def get_status():
    epoch, labels = query_prometheus("training_epoch")
    batch, _ = query_prometheus("training_batch")
    images, _ = query_prometheus("training_images_processed")
    loss, _ = query_prometheus("training_loss")
    auc, _ = query_prometheus("training_auc")
    val_loss, _ = query_prometheus("validation_loss")
    val_auc, _ = query_prometheus("validation_auc")

    if epoch is None:
        return (
            "⚪ Chest X-ray Training\n\n"
            "ยังไม่พบ Training Metrics\n"
            "โมเดลอาจยังไม่เริ่ม Train"
        )

    model = labels.get("model", "Unknown") if labels else "Unknown"

    lines = [
        "🧠 Chest X-ray Training Status",
        "",
        f"🟢 Model: {model}",
        f"Epoch: {int(epoch)}",
    ]

    if batch is not None:
        lines.append(f"Batch: {int(batch)}")

    if images is not None:
        lines.append(f"Images processed: {int(images):,}")

    lines.append("")

    if loss is not None:
        lines.append(f"Loss: {loss:.4f}")

    if auc is not None:
        lines.append(f"AUC: {auc:.4f}")

    if val_loss is not None:
        lines.append(f"Val Loss: {val_loss:.4f}")

    if val_auc is not None:
        lines.append(f"Val AUC: {val_auc:.4f}")

    return "\n".join(lines)


def handle_message(message):
    chat_id = message["chat"]["id"]
    text = message.get("text", "").strip()

    if text == "/status":
        send_message(chat_id, get_status())

    elif text == "/metrics":
        send_message(chat_id, get_status())

    elif text == "/help":
        send_message(
            chat_id,
            "📋 Commands\n\n"
            "/status - ดูสถานะ Training\n"
            "/metrics - ดูค่า Loss / AUC\n"
            "/help - ดูคำสั่ง"
        )

    else:
        send_message(
            chat_id,
            "ใช้คำสั่ง /status เพื่อดูสถานะ Training"
        )


def main():
    print("🤖 Telegram monitoring bot started")

    offset = None

    while True:
        try:
            params = {
                "timeout": 30
            }

            if offset is not None:
                params["offset"] = offset

            r = requests.get(
                f"{BASE_URL}/getUpdates",
                params=params,
                timeout=35
            )

            updates = r.json().get("result", [])

            for update in updates:
                offset = update["update_id"] + 1

                if "message" in update:
                    handle_message(update["message"])

        except Exception as e:
            print("Telegram error:", e)
            time.sleep(5)


if __name__ == "__main__":
    main()