import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warning logs

import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import tensorflow as tf
import pytesseract
import re
import cv2  # Added for image preprocessing

app = Flask(__name__)

# Load models
model1 = tf.keras.models.load_model('models/model1.h5')
model2 = tf.keras.models.load_model('models/model2.h5')
model3 = tf.keras.models.load_model('models/model3.h5')
model4 = tf.keras.models.load_model('models/model4.h5')

# Class label mapping
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

# Enhanced text pattern rules with confidence scoring
classification_rules = {
    'model2': {
        'China': [
            {'pattern': r'P\.R\. China', 'confidence': 0.95},
            {'pattern': r'Model: MXR-310A', 'confidence': 0.85},
            {'pattern': r'Shunde Huierjia Electrical', 'confidence': 0.90}
        ],
        'Japan': [
            {'pattern': r'Made in Japan', 'confidence': 0.95},
            {'pattern': r'NMZH Electronics', 'confidence': 0.85},
            {'pattern': r'DRV-521M', 'confidence': 0.80}
        ],
        'Malaysia': [
            {'pattern': r'Made in Malaysia', 'confidence': 0.95},
            {'pattern': r'SIRIM', 'confidence': 0.85}
        ],
        'UK': [
            {'pattern': r'Made in (the )?UK', 'confidence': 0.97},
            {'pattern': r'Type RFX-200X', 'confidence': 0.90},
            {'pattern': r'Product of UK', 'confidence': 0.85},
            {'pattern': r'UK origin', 'confidence': 0.80}
        ]
    }
}

def preprocess_image_for_text(img_pil):
    """Enhanced image preprocessing for OCR"""
    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # Advanced preprocessing pipeline
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Noise removal and enhancement
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    processed = cv2.medianBlur(processed, 3)
    
    # Convert back to PIL
    return Image.fromarray(processed)

def extract_text_from_image(img_pil):
    """Robust text extraction with error correction"""
    processed_img = preprocess_image_for_text(img_pil)
    
    # Configure Tesseract for multiple languages
    custom_config = r'--oem 3 --psm 6 -l eng+chi_sim+jpn'
    text = pytesseract.image_to_string(processed_img, config=custom_config)
    
    # Common OCR error corrections specific to power labels
    corrections = {
        'rnade': 'made',
        'the uk': 'uk',
        'u.k': 'uk',
        'type rfx-200x': 'type rfx-200x',
        'power uk': 'poweruk',
        'power china': 'powerchina'
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    return text.lower()

def classify_by_text(text, model_name):
    """Enhanced text classification with confidence scoring"""
    if model_name not in classification_rules:
        return None
    
    best_match = {'country': None, 'confidence': 0, 'pattern': None}
    
    for country, patterns in classification_rules[model_name].items():
        for pattern_info in patterns:
            if re.search(pattern_info['pattern'], text, re.IGNORECASE):
                if pattern_info['confidence'] > best_match['confidence']:
                    best_match = {
                        'country': country,
                        'confidence': pattern_info['confidence'],
                        'pattern': pattern_info['pattern']
                    }
    
    return best_match if best_match['country'] else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data.get('image')
        model_name = data.get('model_name')

        if not image_data or not model_name:
            return jsonify({'error': 'Image or model_name missing'}), 400

        if len(image_data) < 1000:
            return jsonify({'error': 'Camera feed not ready or image is empty'}), 400
        
        # Process image
        img_data = base64.b64decode(image_data.split(',')[1])
        img_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
        
        # Check if image is too dark
        if np.mean(img_pil) < 0.05:
            return jsonify({'error': 'Image too dark or camera not working'}), 400

        # Model-specific processing
        if model_name == 'model2':  # POWER - Hybrid text/image classification
            # First try text classification
            extracted_text = extract_text_from_image(img_pil)
            print(f"Extracted Text: {extracted_text}")
            
            text_result = classify_by_text(extracted_text, model_name)
            if text_result:
                return jsonify({
                    'model': model_name,
                    'prediction': text_result['country'],
                    'method': 'text',
                    'matched_pattern': text_result['pattern'],
                    'confidence': text_result['confidence']
                })
            
            # Fallback to image classification if text fails
            img = img_pil.resize((224, 224))
            img_array = np.asarray(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            model = model_mapping.get(model_name)
            prediction = model.predict(img_array)
            index = int(np.argmax(prediction[0]))
            label = class_names[model_name][index]
            
            return jsonify({
                'model': model_name,
                'prediction': label,
                'method': 'image',
                'confidence': float(np.max(prediction[0]))
            })
        else:  # Image-based classification for other models
            img = img_pil.resize((224, 224))
            img_array = np.asarray(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            model = model_mapping.get(model_name)
            prediction = model.predict(img_array)
            index = int(np.argmax(prediction[0]))
            label = class_names[model_name][index]
            
            return jsonify({
                'model': model_name,
                'prediction': label,
                'method': 'image',
                'confidence': float(np.max(prediction[0]))
            })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/load_model/<model_name>', methods=['GET'])
def load_model_route(model_name):
    if model_name in model_mapping:
        return jsonify({"status": "success", "model": model_name})
    return jsonify({"status": "error", "message": "Model not found!"}), 404

if __name__ == "__main__":
    app.run(debug=True)