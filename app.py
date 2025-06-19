from flask import Flask, request, jsonify
from PIL import Image
import io
import time
import requests
from ultralytics import YOLO

app = Flask(__name__)

# === CONFIG ===
TELEGRAM_BOT_TOKEN = '7251920125:AAEtmF2MJsSZ-x0MkssHpksCRuEtZsn9TvI'
TELEGRAM_CHAT_ID = '1587098318'
DETECTION_INTERVAL = 10  # Minimum seconds between two alerts

# === Load YOLOv5 model ===
model = YOLO("yolov5s.pt")  # Include this file in your Render repo
last_alert_time = 0

# === Telegram alert ===
def send_telegram_alert():
    global last_alert_time
    now = time.time()
    if now - last_alert_time >= DETECTION_INTERVAL:
        message = "🚨 Human detected by ESP32-CAM AI!"
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        try:
            response = requests.post(url, data=data)
            if response.status_code == 200:
                print("[INFO] Telegram alert sent")
            else:
                print(f"[ERROR] Telegram failed: {response.text}")
            last_alert_time = now
        except Exception as e:
            print(f"[ERROR] Telegram exception: {e}")

# === Upload endpoint for ESP32-CAM ===
@app.route('/upload', methods=['POST'])
def upload():
    try:
        img_bytes = request.data
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        results = model(img)
        people_count = 0
        for r in results:
            people_count += sum([1 for c in r.boxes.cls if int(c) == 0])  # class 0 = person

        if people_count > 0:
            send_telegram_alert()

        return jsonify({
            "status": "success",
            "people_detected": people_count
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# === Home Page ===
@app.route('/')
def index():
    return "<h2>ESP32-CAM AI Detection Server is Running</h2>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
