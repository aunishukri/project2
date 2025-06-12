from flask import Flask, render_template, send_file, Response, jsonify
import pandas as pd
import os
import cv2
from picamera2 import Picamera2
from pyzbar import pyzbar
import time

app = Flask(__name__)

# Constants
METADATA_FILE = "barcode_metadata.csv"
BARCODE_FILE = "last_barcode.txt"
SNAPSHOT_FILE = "last_frame.jpg"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_STABLE_FRAMES = 5

# Load metadata
if not os.path.exists(METADATA_FILE):
    raise FileNotFoundError("barcode_metadata.csv not found")
metadata_df = pd.read_csv(METADATA_FILE, dtype={'barcode_id': str})

# Camera setup
camera = Picamera2()
config = camera.create_preview_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
)
camera.configure(config)
camera.start()

# State
scanning = False
current_barcode = None
last_barcode = None
stable_count = 0

def generate_frames():
    global scanning, current_barcode, last_barcode, stable_count

    while True:
        frame = camera.capture_array()

        if scanning:
            barcodes = pyzbar.decode(frame)

            if barcodes:
                barcode = barcodes[0]
                barcode_data = barcode.data.decode("utf-8").strip()
                (x, y, w, h) = barcode.rect

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, barcode_data, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if barcode_data == last_barcode:
                    stable_count += 1
                else:
                    stable_count = 1
                    last_barcode = barcode_data

                if stable_count >= MAX_STABLE_FRAMES:
                    current_barcode = barcode_data
                    cv2.imwrite(SNAPSHOT_FILE, frame)
                    with open(BARCODE_FILE, "w") as f:
                        f.write(barcode_data)
                    scanning = False
            else:
                stable_count = 0
                last_barcode = None

        # Convert to JPEG and yield
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start_scan")
def start_scan():
    global scanning, current_barcode, stable_count
    scanning = True
    current_barcode = None
    stable_count = 0

    # Clear last scan result
    if os.path.exists(BARCODE_FILE):
        os.remove(BARCODE_FILE)
    if os.path.exists(SNAPSHOT_FILE):
        os.remove(SNAPSHOT_FILE)

    return "", 200

@app.route("/barcode_data")
def barcode_data():
    barcode = None
    barcode_info = None
    image_exists = os.path.exists(SNAPSHOT_FILE)

    if os.path.exists(BARCODE_FILE):
        with open(BARCODE_FILE, "r") as f:
            barcode = f.read().strip()
            if barcode:
                match = metadata_df[metadata_df['barcode_id'] == barcode]
                if not match.empty:
                    barcode_info = match.iloc[0].to_dict()

    return jsonify({
        "barcode": barcode,
        "barcode_info": barcode_info,
        "image_exists": image_exists
    })

@app.route("/image")
def image():
    if os.path.exists(SNAPSHOT_FILE):
        return send_file(SNAPSHOT_FILE, mimetype='image/jpeg')
    else:
        return "", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)