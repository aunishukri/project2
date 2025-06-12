import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import tensorflow as tf
import pytesseract
import re
import traceback
import cv2
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


app = Flask(__name__)
# ✅ MongoDB connection
# Replace with your actual MongoDB URI
mongo_uri = "mongodb+srv://auniathirah0096:Myself102705@cluster0.fdmy3yk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri, server_api=ServerApi('1'))
DATABASE_NAME = 'prediction'
COLLECTION_NAME = 'status'
db = client[DATABASE_NAME]


# ✅ Load models
model1 = tf.keras.models.load_model('models/model1.h5')
model2 = tf.keras.models.load_model('models/model2.h5')
model3 = tf.keras.models.load_model('models/model3.h5')
model4 = tf.keras.models.load_model('models/model4.h5')

# ✅ Label mapping
class_names = {
    'model1': ['NG', 'OK'],
    'model2': ['China', 'Japan', 'Malaysia', 'UK'],
    'model3': ['China', 'Japan', 'Malaysia', 'UK'],
    'model4': ['CCC', 'PSE', 'SIRIM', 'UKNA']
}

model_mapping = {
    'model1': model1,
    'model2': model2,
    'model3': model3,
    'model4': model4
}

# ✅ Text-matching rules
classification_rules = {
    'model2': {
        'China': [{'pattern': r'P\.R\. China', 'confidence': 0.95}, {'pattern': r'Model: MXR-310A', 'confidence': 0.85}, {'pattern': r'Shunde Huierjia Electrical', 'confidence': 0.90}],
        'Japan': [{'pattern': r'MADE IN JAPAN', 'confidence': 0.95}, {'pattern': r'NMZH Electronics Co., LTD', 'confidence': 0.85}, {'pattern': r'DRV-521M', 'confidence': 0.90}, {'pattern': r'型号名称-制造编号', 'confidence': 0.80}],
        'Malaysia': [{'pattern': r'MADE IN MALAYSIA', 'confidence': 0.95}],
        'UK': [{'pattern': r'Made in the UK', 'confidence': 0.95}, {'pattern': r'Type RFX-200X', 'confidence': 0.85}, {'pattern': r'240V 50/60Hz', 'confidence': 0.90}]
    },
    'model3': {
        'China': [{'pattern': r'黑白', 'confidence': 0.90}, {'pattern': r'启动', 'confidence': 0.90}, {'pattern': r'彩色', 'confidence': 0.90}, {'pattern': r'停止', 'confidence': 0.90}],
        'Japan': [{'pattern': r'白黒', 'confidence': 0.90}, {'pattern': r'開始', 'confidence': 0.85}, {'pattern': r'カラー', 'confidence': 0.90}, {'pattern': r'一時停止', 'confidence': 0.90}],
        'Malaysia': [{'pattern': r'Malaysia', 'confidence': 0.95}],
        'UK': [{'pattern': r'Black', 'confidence': 0.90}, {'pattern': r'Colour', 'confidence': 0.90}, {'pattern': r'Start', 'confidence': 0.90}, {'pattern': r'Stop', 'confidence': 0.90}]
    }
}

# ✅ Load metadata
barcode_df = pd.read_csv('barcode_metadata.csv')
barcode_df.set_index('barcode_id', inplace=True)

# ✅ Function untuk semak jangkaan berdasarkan barcode & model
def get_expected_prediction(barcode, model_name):
    try:
        return barcode_df.loc[barcode, model_name]
    except KeyError:
        return None

# ✅ Function baca current barcode
def get_current_barcode():
    try:
        with open("current_barcode.txt", "r") as f:
            return f.read().strip()
    except:
        return "UNKNOWN"
    
# Save prediction result to MongoDB
def save_prediction_to_mongo(barcode, model_name, prediction, method, confidence, status):
    try:
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        doc = {
            'barcode': barcode,
            'model': model_name,
            'prediction': prediction,
            'method': method,
            'confidence': confidence,
            'status': status
        }
        collection.insert_one(doc)
        print(f"[MongoDB] {model_name} result saved.")
    except Exception as e:
        print(f"[MongoDB Error] {e}")

def preprocess_image_for_text(img):
    img = img.convert('L')
    img = img.point(lambda x: 0 if x < 128 else 255)
    return img

def extract_text_from_image(img):
    open_cv_image = np.array(img.convert("RGB"))[:, :, ::-1].copy()
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 10:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    extracted = cv2.bitwise_and(gray, gray, mask=mask)
    custom_config = r'--oem 3 --psm 6 -l eng+chi_sim+jpn'
    text = pytesseract.image_to_string(Image.fromarray(extracted), config=custom_config).strip()
    text = text.strip()
    corrections = {
        'rnade': 'Made', 'the uk': 'the Uk', 'u.k': 'UK', 'type rfx-200x': 'Type RFX-200X'
    }
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text

def classify_country_by_text(extracted_text, rules):
    for country, patterns in rules.items():
        for rule in patterns:
            if re.search(rule['pattern'], extracted_text, re.IGNORECASE):
                print(f"[DEBUG] Match found: {rule['pattern']} => {country}")
                return country, rule['confidence']
    return 'Unknown', 0.0
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data.get('image')
        model_name = data.get('model_name')

        if not image_data or not model_name or len(image_data) < 1000:
            return jsonify({'error': 'Invalid image or model name'}), 400

        img_data = base64.b64decode(image_data.split(',')[1])
        img_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
        img = img_pil.resize((224, 224))
        img_array = np.asarray(img, dtype=np.float32) / 255.0

        if np.mean(img_array) < 0.05:
            return jsonify({'error': 'Image too dark or camera not working'}), 400

        current_barcode = get_current_barcode()
        expected = get_expected_prediction(current_barcode, model_name)
        
        if model_name in ['model2', 'model3']:
            text_img = preprocess_image_for_text(img_pil)
            extracted_text = extract_text_from_image(text_img)
            print(f"[OCR-{model_name}] Extracted Text: {extracted_text}")
            rules = classification_rules[model_name]
            country, confidence = classify_country_by_text(extracted_text, rules)
            if country != 'Unknown':
                status = "OK" if expected and country.strip().lower() == str(expected).strip().lower() else "NG"
                return jsonify({
                    'model': model_name,
                    'prediction': country,
                    'method': 'text',
                    'confidence': confidence,
                    'status': status
                })

        # Image fallback
        model = model_mapping.get(model_name)
        labels = class_names.get(model_name)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        index = int(np.argmax(prediction[0]))
        label = labels[index]
        confidence = float(np.max(prediction[0]))

        
        status = "OK" if expected and label.strip().lower() == str(expected).strip().lower() else "NG"
        save_prediction_to_mongo(current_barcode, model_name, label, 'image', confidence, status)

        
        return jsonify({
            'model': model_name,
            'prediction': label,
            'method': 'image',
            'confidence': float(np.max(prediction[0])),
            'status': status
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/barcode')
def get_barcode():
    try:
        with open("current_barcode.txt", "r") as f:
            barcode = f.read().strip()
        return jsonify({'barcode': barcode})
    except:
        return jsonify({'barcode': 'No barcode'})


@app.route('/load_model/<model_name>', methods=['GET'])
def load_model_route(model_name):
    if model_name in model_mapping:
        return jsonify({"status": "success", "model": model_name})
    return jsonify({"status": "error", "message": "Model not found!"}), 404

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(f"[MongoDB Error] {e}")

if __name__ == "__main__":
    #app.run(debug=True, use_reloader=False) 
   
   app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)  # Disable reloader