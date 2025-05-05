"""
Improved Web-based GUI for Waste Classification with Object Detection
Author: Zeyad-Diaa-1242
Last updated: 2025-05-05 21:04:05
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import threading
import webbrowser
import time
import pickle
import gc
from feature_extraction import FeatureExtractor

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
MODEL_PATHS = [
    './model_checkpoints/waste_model_best.h5',  # First choice
    'waste_classifier_anti_overfitting.h5'      # Fallback
]
CLASSES = ['Glass', 'Metal', 'Paper', 'Plastic']
DISPLAY_CLASSES = ['Glass', 'Metal', 'Paper', 'Plastic']  # Classes to actually show

# Get the current date and username for display
current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
current_user = os.getenv('USER', 'Zeyad-Diaa-1242')

# Load components
print("Loading classification components...")

# 1. Load model
model = None
for model_path in MODEL_PATHS:
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Loaded classification model from {model_path}")
            break
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")

if model is None:
    raise RuntimeError("Could not load any classification model!")

# 2. Load feature scaler
scaler = None
scaler_paths = ['exported_models/feature_scaler.pkl', 'feature_scaler.pkl']
for scaler_path in scaler_paths:
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"Loaded feature scaler from {scaler_path}")
            break
        except Exception as e:
            print(f"Error loading scaler from {scaler_path}: {e}")

# If scaler is not found, create a default one
if scaler is None:
    print("Creating default feature scaler")
    scaler = StandardScaler()

# 3. Initialize feature extractor exactly as during training
feature_extractor = FeatureExtractor(input_shape=(224, 224, 3))
print("Initialized feature extractor")

# Object detection model - we'll use a simple OpenCV-based approach for segmentation
# but log key information about the process

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(file_path):
    """Preprocess uploaded image exactly like in training"""
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Could not read image from {file_path}")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def detect_objects(img, debug_info=None):
    """Advanced object detection for waste items using multiple methods"""
    if debug_info is None:
        debug_info = []
    
    debug_info.append(f"Input image shape: {img.shape}")
    
    # Original image dimensions
    original_h, original_w = img.shape[:2]
    
    # Resize for processing if needed
    working_img = img
    if img.shape[0] > 640 or img.shape[1] > 640:
        working_img = cv2.resize(img, (640, 480))
        debug_info.append(f"Resized image for processing: {working_img.shape}")
    
    all_boxes = []
    
    # Method 1: Color-based segmentation
    debug_info.append("Method 1: Color-based segmentation")
    try:
        # Convert to different color spaces
        hsv = cv2.cvtColor(working_img, cv2.COLOR_RGB2HSV)
        
        # Create masks for different colors
        color_masks = []
        
        # Look for common waste material colors
        hsv_ranges = [
            # Name, Lower HSV, Upper HSV
            ("Clear/Glass", [0, 0, 200], [180, 30, 255]),      # Glass/clear materials
            ("Blue", [100, 50, 50], [130, 255, 255]),          # Blue plastic/metal
            ("Green", [40, 50, 50], [80, 255, 255]),           # Green bottles/plastic
            ("Brown", [10, 50, 50], [20, 255, 200]),           # Paper/cardboard
            ("White", [0, 0, 200], [180, 30, 255]),            # Paper/plastic
            ("Gray", [0, 0, 70], [180, 30, 200]),              # Metal
            ("Yellow", [20, 100, 100], [35, 255, 255]),        # Plastic
            ("Red", [0, 100, 100], [10, 255, 255]),            # Plastic
            ("Red2", [170, 100, 100], [180, 255, 255]),        # Red continues at both ends of hue
        ]
        
        for name, lower, upper in hsv_ranges:
            # Create mask for this color range
            lower_arr = np.array(lower)
            upper_arr = np.array(upper)
            mask = cv2.inRange(hsv, lower_arr, upper_arr)
            
            # Find contours in this mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by size
            min_area = working_img.shape[0] * working_img.shape[1] * 0.01  # 1% of image
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    all_boxes.append((x, y, w, h, name))
                    debug_info.append(f"    Found {name} object: x={x}, y={y}, w={w}, h={h}, area={area:.1f}")
        
        debug_info.append(f"    Color-based detection found {len(all_boxes)} objects")
    
    except Exception as e:
        debug_info.append(f"    Error in color-based segmentation: {e}")
    
    # Method 2: Edge-based segmentation
    debug_info.append("Method 2: Edge-based segmentation")
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(working_img, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect them
        kernel = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by size
        min_area = working_img.shape[0] * working_img.shape[1] * 0.01  # 1% of image
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                all_boxes.append((x, y, w, h, "Edge"))
                debug_info.append(f"    Found edge-based object: x={x}, y={y}, w={w}, h={h}, area={area:.1f}")
        
        debug_info.append(f"    Edge-based detection found {len(all_boxes) - len(all_boxes)} additional objects")
    
    except Exception as e:
        debug_info.append(f"    Error in edge-based segmentation: {e}")
    
    # Add full-image box if no objects found or only a few
    if len(all_boxes) < 2:
        debug_info.append("Adding full-image box as fallback")
        h, w = working_img.shape[:2]
        all_boxes.append((0, 0, w, h, "Full"))
    
    # Scale boxes back to original image size if needed
    if working_img.shape[:2] != (original_h, original_w):
        scale_x = original_w / working_img.shape[1]
        scale_y = original_h / working_img.shape[0]
        
        scaled_boxes = []
        for x, y, w, h, label in all_boxes:
            scaled_boxes.append((
                int(x * scale_x),
                int(y * scale_y),
                int(w * scale_x),
                int(h * scale_y),
                label
            ))
        all_boxes = scaled_boxes
        debug_info.append(f"Scaled boxes to original image size (scale_x={scale_x:.2f}, scale_y={scale_y:.2f})")
    
    # Merge overlapping boxes
    if len(all_boxes) > 1:
        debug_info.append(f"Merging overlapping boxes from initial count: {len(all_boxes)}")
        all_boxes = merge_boxes(all_boxes, debug_info)
    
    # Remove the label from the boxes (we only need x,y,w,h)
    boxes = [(x, y, w, h) for x, y, w, h, _ in all_boxes]
    
    debug_info.append(f"Final box count: {len(boxes)}")
    return boxes, debug_info

def merge_boxes(boxes, debug_info=None):
    """Merge boxes with significant overlap"""
    if debug_info is None:
        debug_info = []
    
    if not boxes:
        return []
    
    # Sort boxes by area (largest first)
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    merged = []
    
    while boxes:
        current_box = boxes.pop(0)
        x1, y1, w1, h1, label1 = current_box
        
        # Check overlap with remaining boxes
        i = 0
        while i < len(boxes):
            x2, y2, w2, h2, label2 = boxes[i]
            
            # Calculate intersection area
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            intersection = x_overlap * y_overlap
            
            # Calculate areas
            area1 = w1 * h1
            area2 = w2 * h2
            
            # Calculate IoU (Intersection over Union)
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0
            
            # If significant overlap, merge boxes
            if iou > 0.3:  # IoU threshold
                # Create new merged box
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1 + w1, x2 + w2)
                y_max = max(y1 + h1, y2 + h2)
                
                # Keep the label from the larger box
                merged_label = label1 if area1 > area2 else label2
                
                debug_info.append(f"    Merging boxes: {(x1,y1,w1,h1)} and {(x2,y2,w2,h2)} with IoU={iou:.2f}")
                
                # Update current box to merged box
                x1, y1 = x_min, y_min
                w1, h1 = x_max - x_min, y_max - y_min
                label1 = merged_label
                
                # Remove the merged box
                boxes.pop(i)
            else:
                i += 1
        
        merged.append((x1, y1, w1, h1, label1))
    
    return merged

def classify_regions(img, boxes, debug_info=None):
    """Classify each detected region using the feature extraction pipeline"""
    if debug_info is None:
        debug_info = []
        
    debug_info.append(f"Starting classification of {len(boxes)} regions")
    results = []
    
    # Process each region
    for box_idx, box in enumerate(boxes):
        x, y, w, h = box
        
        # Ensure box is within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        
        # Skip if box is too small
        if w < 20 or h < 20:
            debug_info.append(f"    Box {box_idx} is too small: {box}")
            continue
        
        # Extract region and resize for classification
        region = img[y:y+h, x:x+w]
        resized = cv2.resize(region, (224, 224))
        
        # Extract features using the same feature extractor as training
        try:
            debug_info.append(f"    Extracting features for box {box_idx}: {box}")
            features = feature_extractor.extract_features(resized)
            
            # Replace any NaN or inf values
            features = np.nan_to_num(features)
            
            # Add batch dimension
            features = np.array([features])
            
            # Log feature statistics
            debug_info.append(f"    Feature shape: {features.shape}, Mean: {np.mean(features):.3f}, Std: {np.std(features):.3f}")
            
            # Scale features
            try:
                features_scaled = scaler.transform(features)
                debug_info.append(f"    Scaled features, Mean: {np.mean(features_scaled):.3f}, Std: {np.std(features_scaled):.3f}")
            except Exception as e:
                debug_info.append(f"    Error scaling features: {e}, using unscaled features")
                features_scaled = features
            
            # Get predictions
            predictions = model.predict(features_scaled, verbose=0)[0]
            
            # Get top predictions
            top_indices = np.argsort(predictions)[-3:][::-1]  # Top 3
            
            debug_info.append(f"    Top predictions for box {box_idx}:")
            for idx in top_indices:
                class_name = CLASSES[idx]
                confidence = predictions[idx]
                debug_info.append(f"      {class_name}: {confidence:.3f}")
                
                # Only include target classes
                if class_name in DISPLAY_CLASSES and confidence > 0.2:
                    results.append({
                        'box': box,
                        'class': class_name,
                        'confidence': float(confidence)
                    })
        
        except Exception as e:
            debug_info.append(f"    Error classifying region {box_idx}: {e}")
    
    # If no results or all bad confidence, classify the whole image
    if not results or max((r['confidence'] for r in results), default=0) < 0.3:
        debug_info.append("No confident detections, classifying whole image")
        try:
            # Resize whole image
            whole_img = cv2.resize(img, (224, 224))
            
            # Extract features
            features = feature_extractor.extract_features(whole_img)
            features = np.array([np.nan_to_num(features)])
            
            # Scale features
            try:
                features_scaled = scaler.transform(features)
            except Exception as e:
                debug_info.append(f"Error scaling whole-image features: {e}, using unscaled")
                features_scaled = features
            
            # Get predictions
            predictions = model.predict(features_scaled, verbose=0)[0]
            
            # Get top predictions
            top_indices = np.argsort(predictions)[-2:][::-1]  # Top 2
            
            debug_info.append("Whole image classification results:")
            for idx in top_indices:
                class_name = CLASSES[idx]
                confidence = predictions[idx]
                debug_info.append(f"  {class_name}: {confidence:.3f}")
                
                if class_name in DISPLAY_CLASSES and confidence > 0.2:
                    # Create a central box
                    h, w = img.shape[:2]
                    box_w, box_h = int(w * 0.8), int(h * 0.8)
                    x, y = int((w - box_w) / 2), int((h - box_h) / 2)
                    
                    results.append({
                        'box': (x, y, box_w, box_h),
                        'class': class_name,
                        'confidence': float(confidence)
                    })
        
        except Exception as e:
            debug_info.append(f"Error in whole-image classification: {e}")
    
    debug_info.append(f"Final classification results: {len(results)} objects")
    return results, debug_info

def draw_results(img, results):
    """Draw bounding boxes and labels on the image"""
    img_copy = img.copy()
    
    # Define colors for classes (RGB format)
    colors = {
        'Glass': (0, 255, 0),     # Green
        'Metal': (0, 0, 255),     # Blue 
        'Paper': (255, 165, 0),   # Orange
        'Plastic': (255, 0, 255)  # Magenta
    }
    
    for result in results:
        x, y, w, h = result['box']
        class_name = result['class']
        confidence = result['confidence']
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, img.shape[1]-1))
        y = max(0, min(y, img.shape[0]-1))
        w = min(w, img.shape[1]-x)
        h = min(h, img.shape[0]-y)
        
        # Get color for class
        color = colors.get(class_name, (200, 200, 200))  # Default to gray
        
        # Draw thicker bounding box (3 pixels)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 3)
        
        # Prepare label text
        label = f"{class_name}: {confidence:.2f}"
        
        # Calculate label position
        label_y = max(y - 10, 20)
        
        # Draw filled background for text
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_copy, (x, label_y - text_size[1] - 5), 
                     (x + text_size[0], label_y + 5), color, -1)
        
        # Draw label text in white
        cv2.putText(img_copy, label, (x, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img_copy

def img_to_base64(img):
    """Convert OpenCV image to base64 string for web display"""
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

# Flask routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', 
                          current_date=current_date, 
                          current_user=current_user)

@app.route('/upload', methods=['POST'])
def upload():
    """Handle image upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        try:
            debug_info = [
                f"Processing started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Processing image file: {filename}"
            ]
            
            # Load and preprocess image
            img = preprocess_image(filepath)
            debug_info.append(f"Loaded image with shape: {img.shape}")
            
            # Detect objects in the image
            boxes, debug_info = detect_objects(img, debug_info)
            
            # Classify each detected region
            results, debug_info = classify_regions(img, boxes, debug_info)
            
            # Draw results on image
            result_img = draw_results(img, results)
            
            # Convert to base64 for display
            img_base64 = img_to_base64(result_img)
            
            # Prepare result data
            detection_data = []
            for result in results:
                detection_data.append({
                    'class': result['class'],
                    'confidence': round(result['confidence'] * 100, 1),
                    'box': result['box']
                })
            
            debug_info.append(f"Processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return jsonify({
                'success': True,
                'img_data': img_base64,
                'detections': detection_data,
                'debug_info': '\n'.join(debug_info)
            })
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            return jsonify({
                'error': f'Error processing image: {str(e)}',
                'traceback': error_traceback
            }), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Create templates folder and index.html
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w') as f:
    f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Advanced Waste Classification</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 960px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f7f9fc;
            color: #333;
        }}
        header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: #7f8c8d;
            margin-top: 0;
        }}
        .container {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 25px;
            margin-bottom: 25px;
        }}
        .upload-container {{
            text-align: center;
        }}
        .file-input-wrapper {{
            position: relative;
            margin: 20px auto;
            max-width: 400px;
        }}
        .file-input {{
            opacity: 0;
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }}
        .file-input-label {{
            display: block;
            padding: 15px;
            background-color: #e8f4fd;
            border: 2px dashed #3498db;
            border-radius: 6px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }}
        .file-input-label:hover {{
            background-color: #d6eafc;
            border-color: #2980b9;
        }}
        .file-input-icon {{
            display: block;
            font-size: 24px;
            margin-bottom: 10px;
            color: #3498db;
        }}
        .selected-file {{
            margin-top: 10px;
            font-style: italic;
            color: #7f8c8d;
        }}
        .button {{
            background-color: #3498db;
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 15px;
            transition: background-color 0.3s;
        }}
        .button:hover {{
            background-color: #2980b9;
        }}
        .button:disabled {{
            background-color: #bdc3c7;
            cursor: not-allowed;
        }}
        .loading {{
            text-align: center;
            display: none;
            margin: 20px 0;
        }}
        .spinner {{
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 36px;
            height: 36px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .result-container {{
            display: none;
        }}
        .result-image-wrapper {{
            text-align: center;
            margin: 20px 0;
            border: 1px solid #eee;
            padding: 10px;
            border-radius: 8px;
            background-color: #f9f9f9;
        }}
        #result-image {{
            max-width: 100%;
            max-height: 600px;
            margin: 0 auto;
        }}
        .detections-heading {{
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
            margin-top: 30px;
        }}
        .detection-list {{
            list-style-type: none;
            padding: 0;
        }}
        .detection-item {{
            padding: 12px 15px;
            margin: 8px 0;
            border-radius: 6px;
            color: white;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .detection-item .confidence {{
            background-color: rgba(255, 255, 255, 0.2);
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.9em;
        }}
        .Glass {{ background-color: #2ecc71; }}
        .Metal {{ background-color: #3498db; }}
        .Paper {{ background-color: #e67e22; }}
        .Plastic {{ background-color: #9b59b6; }}
        .legend {{
            display: flex;
            justify-content: center;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 15px;
        }}
        .legend-color {{
            width: 24px;
            height: 24px;
            margin-right: 8px;
            border-radius: 4px;
        }}
        .debug-info {{
            font-family: monospace;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 12px;
            margin-top: 25px;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            display: none;
        }}
        .toggle-debug {{
            background: none;
            border: none;
            color: #7f8c8d;
            text-decoration: underline;
            cursor: pointer;
            font-size: 0.9em;
            margin-top: 15px;
            display: block;
            text-align: center;
        }}
        .toggle-debug:hover {{
            color: #3498db;
        }}
        footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Advanced Waste Classification</h1>
        <p class="subtitle">Upload an image to classify waste materials</p>
    </header>
    
    <div class="container upload-container">
        <h2>Image Upload</h2>
        <form id="upload-form" enctype="multipart/form-data">
            <div class="file-input-wrapper">
                <label class="file-input-label">
                    <span class="file-input-icon">📷</span>
                    <span id="file-prompt">Drag & drop an image or click to browse</span>
                    <input type="file" id="file-input" name="file" accept=".jpg, .jpeg, .png" class="file-input">
                </label>
            </div>
            <div class="selected-file" id="selected-file"></div>
            <button type="submit" class="button" id="submit-button" disabled>Classify Waste</button>
        </form>
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing image and detecting waste objects...</p>
        </div>
    </div>
    
    <div class="container result-container" id="result-container">
        <h2>Classification Results</h2>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color Glass"></div>
                <span>Glass</span>
            </div>
            <div class="legend-item">
                <div class="legend-color Metal"></div>
                <span>Metal</span>
            </div>
            <div class="legend-item">
                <div class="legend-color Paper"></div>
                <span>Paper</span>
            </div>
            <div class="legend-item">
                <div class="legend-color Plastic"></div>
                <span>Plastic</span>
            </div>
        </div>
        
        <div class="result-image-wrapper">
            <img id="result-image" src="" alt="Classification result">
        </div>
        
        <h3 class="detections-heading">Detected Items</h3>
        <ul class="detection-list" id="detection-list">
            <!-- Detections will be added here -->
        </ul>
        
        <button class="toggle-debug" id="toggle-debug">Show Technical Details</button>
        <div class="debug-info" id="debug-info">
            <!-- Debug info will be added here -->
        </div>
    </div>
    
    <footer>
        <p>User: {current_user} | Date: {current_date}</p>
        <p>Waste Classification Model v2.0 | Using Feature Extraction Pipeline</p>
    </footer>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const fileInput = document.getElementById('file-input');
            const filePrompt = document.getElementById('file-prompt');
            const selectedFile = document.getElementById('selected-file');
            const submitButton = document.getElementById('submit-button');
            const toggleDebug = document.getElementById('toggle-debug');
            const debugInfo = document.getElementById('debug-info');
            
            // Handle file selection
            fileInput.addEventListener('change', function() {
                if (fileInput.files.length > 0) {
                    const fileName = fileInput.files[0].name;
                    selectedFile.textContent = `Selected file: ${fileName}`;
                    submitButton.disabled = false;
                } else {
                    selectedFile.textContent = '';
                    submitButton.disabled = true;
                }
            });
            
            // Toggle debug info
            toggleDebug.addEventListener('click', function() {
                if (debugInfo.style.display === 'none' || !debugInfo.style.display) {
                    debugInfo.style.display = 'block';
                    toggleDebug.textContent = 'Hide Technical Details';
                } else {
                    debugInfo.style.display = 'none';
                    toggleDebug.textContent = 'Show Technical Details';
                }
            });
            
            // Handle form submission
            const form = document.getElementById('upload-form');
            form.addEventListener('submit', function(e) {
                e.preventDefault();
                
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select an image file');
                    return;
                }
                
                // Show loading spinner
                document.getElementById('loading').style.display = 'block';
                submitButton.disabled = true;
                
                // Create FormData and send request
                const formData = new FormData();
                formData.append('file', file);
                
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading spinner
                    document.getElementById('loading').style.display = 'none';
                    submitButton.disabled = false;
                    
                    if (data.success) {
                        // Show results container
                        document.getElementById('result-container').style.display = 'block';
                        
                        // Set result image
                        document.getElementById('result-image').src = 'data:image/png;base64,' + data.img_data;
                        
                        // Add detections to list
                        const detectionList = document.getElementById('detection-list');
                        detectionList.innerHTML = '';
                        
                        if (data.detections.length === 0) {
                            detectionList.innerHTML = '<li>No waste items detected</li>';
                        } else {
                            data.detections.forEach(detection => {
                                const listItem = document.createElement('li');
                                listItem.className = `detection-item ${detection.class}`;
                                
                                const nameSpan = document.createElement('span');
                                nameSpan.textContent = detection.class;
                                
                                const confidenceSpan = document.createElement('span');
                                confidenceSpan.className = 'confidence';
                                confidenceSpan.textContent = `${detection.confidence}% confidence`;
                                
                                listItem.appendChild(nameSpan);
                                listItem.appendChild(confidenceSpan);
                                
                                detectionList.appendChild(listItem);
                            });
                        }
                        
                        // Add debug info
                        const debugInfo = document.getElementById('debug-info');
                        debugInfo.textContent = data.debug_info || 'No debug information available';
                        
                        // Scroll to results
                        document.getElementById('result-container').scrollIntoView({
                            behavior: 'smooth'
                        });
                    } else {
                        alert('Error: ' + data.error);
                        console.error(data.traceback);
                    }
                })
                .catch(error => {
                    document.getElementById('loading').style.display = 'none';
                    submitButton.disabled = false;
                    alert('Error processing image: ' + error);
                });
            });
        });
    </script>
</body>
</html>""")

def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    print("\n=== Advanced Waste Classification GUI ===")
    print(f"User: {current_user}")
    print(f"Date: {current_date}")
    print("\nStarting web interface...")
    print("Access the interface at http://127.0.0.1:5000")
    
    # Start browser in a separate thread
    threading.Thread(target=open_browser).start()
    
    # Start Flask app
    app.run(debug=False, port=5000)
