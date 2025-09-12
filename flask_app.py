from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import os
import pickle
import numpy as np
import subprocess
import platform
from datetime import datetime
from sklearn.preprocessing import normalize
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import threading
import time
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# ---------------- CONFIG ----------------
DATA_DIR = "data"
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.pkl")
ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# Global variables for camera and state management
camera = None
current_mode = None  # 'enrollment' or 'attendance'
enrollment_data = {}
attendance_data = set()

# ---------------- MODEL LOADING ----------------
def load_model(model_name="ResNet50"):
    """
    Load embedding model. Can be extended for other backbones.
    """
    if model_name == "ResNet50":
        return ResNet50(weights="imagenet", include_top=False, pooling="avg")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

model = load_model("ResNet50")

# ---------------- CAMERA UTILS (Browser-based) ----------------
# Camera access is now handled by the browser using getUserMedia API

# ---------------- FACE UTILS ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.3, 5)

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    arr = img_to_array(face_img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)

def get_embedding(face_img):
    arr = preprocess_face(face_img)
    embedding = model.predict(arr, verbose=0)[0]
    return normalize([embedding])[0]

# ---------------- DATA STORAGE ----------------
def load_embeddings():
    return pickle.load(open(EMBEDDINGS_FILE, "rb")) if os.path.exists(EMBEDDINGS_FILE) else {}

def save_embeddings(embeddings):
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)

# ---------------- IMAGE PROCESSING ----------------
def process_image_data(image_data, mode):
    """Process base64 image data from browser camera"""
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if mode == 'enrollment':
            return process_enrollment_frame(frame)
        elif mode == 'attendance':
            return process_attendance_frame(frame)
        
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def process_enrollment_frame(frame):
    global enrollment_data
    
    faces = detect_faces(frame)
    result = {'faces': len(faces), 'samples': len(enrollment_data.get('embeddings', []))}
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        embedding = get_embedding(face)
        
        # Store embedding for current student
        student_id = enrollment_data.get('student_id')
        if student_id:
            if 'embeddings' not in enrollment_data:
                enrollment_data['embeddings'] = []
            enrollment_data['embeddings'].append(embedding)
            result['samples'] = len(enrollment_data['embeddings'])
            
    return result

def process_attendance_frame(frame):
    global attendance_data
    
    embeddings = load_embeddings()
    if not embeddings:
        return {'faces': 0, 'recognized': []}
    
    faces = detect_faces(frame)
    recognized_faces = []
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        emb = get_embedding(face)
        
        # Compare embeddings
        best_match, best_dist = "Unknown", float("inf")
        for sid, stored_emb in embeddings.items():
            dist = np.linalg.norm(stored_emb - emb)
            if dist < best_dist:
                best_match, best_dist = sid, dist
        
        threshold = 0.6
        label = best_match if best_dist < threshold else "Unknown"
        if label != "Unknown":
            attendance_data.add(label)
            recognized_faces.append(label)
        
    return {'faces': len(faces), 'recognized': recognized_faces}

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global current_mode
    
    data = request.json or {}
    image_data = data.get('image')
    mode = data.get('mode', 'attendance')
    
    if not image_data:
        return jsonify({'status': 'error', 'message': 'No image data provided'})
    
    result = process_image_data(image_data, mode)
    
    if result is None:
        return jsonify({'status': 'error', 'message': 'Failed to process image'})
    
    return jsonify({'status': 'success', 'result': result})

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global current_mode
    
    data = request.json or {}
    mode = data.get('mode', 'attendance')
    current_mode = mode
    
    return jsonify({'status': 'success', 'message': f'Camera started in {mode} mode'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global current_mode
    current_mode = None
    return jsonify({'status': 'success', 'message': 'Camera stopped'})

@app.route('/start_enrollment', methods=['POST'])
def start_enrollment():
    global enrollment_data
    
    data = request.json or {}
    student_id = data.get('student_id', '').strip()
    
    if not student_id:
        return jsonify({'status': 'error', 'message': 'Student ID is required'})
    
    enrollment_data = {'student_id': student_id, 'embeddings': []}
    
    return jsonify({'status': 'success', 'message': f'Started enrollment for {student_id}'})

@app.route('/complete_enrollment', methods=['POST'])
def complete_enrollment():
    global enrollment_data
    
    if 'student_id' not in enrollment_data or not enrollment_data.get('embeddings'):
        return jsonify({'status': 'error', 'message': 'No enrollment data found'})
    
    student_id = enrollment_data['student_id']
    embeddings_list = enrollment_data['embeddings']
    
    # Calculate mean embedding
    mean_embedding = np.mean(embeddings_list, axis=0)
    
    # Save to file
    embeddings = load_embeddings()
    embeddings[student_id] = mean_embedding
    save_embeddings(embeddings)
    
    # Clear enrollment data
    enrollment_data = {}
    
    return jsonify({
        'status': 'success', 
        'message': f'Enrolled {student_id} with {len(embeddings_list)} samples'
    })

@app.route('/get_attendance', methods=['GET'])
def get_attendance():
    global attendance_data
    
    attendance_list = list(attendance_data)
    return jsonify({'attendance': attendance_list})

@app.route('/save_attendance', methods=['POST'])
def save_attendance():
    global attendance_data
    
    if not attendance_data:
        return jsonify({'status': 'error', 'message': 'No attendance data to save'})
    
    # Save to CSV
    with open(ATTENDANCE_FILE, "a") as f:
        for sid in attendance_data:
            f.write(f"{sid},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    saved_count = len(attendance_data)
    attendance_data.clear()
    
    return jsonify({
        'status': 'success', 
        'message': f'Saved attendance for {saved_count} students'
    })

@app.route('/clear_attendance', methods=['POST'])
def clear_attendance():
    global attendance_data
    attendance_data.clear()
    return jsonify({'status': 'success', 'message': 'Attendance data cleared'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)