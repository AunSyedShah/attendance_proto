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
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import threading
import time
import base64
import io
from PIL import Image

# Optimize TensorFlow settings to reduce retracing warnings
tf.config.run_functions_eagerly(False)
tf.config.optimizer.set_jit(True)  # Enable XLA compilation for better performance

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

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
    Load embedding model with optimization. Can be extended for other backbones.
    """
    if model_name == "ResNet50":
        # Load model
        model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        
        # Compile with fixed input shape to avoid retracing
        model.compile(optimizer='adam', run_eagerly=False)
        
        # Warm up the model with a dummy prediction to avoid first-call overhead
        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = model.predict(dummy_input, verbose=0)
        
        print("ResNet50 model loaded and optimized")
        return model
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# Load model with optimization
model = load_model("ResNet50")

# ---------------- FACE UTILS ----------------
# Load OpenCV DNN face detection model
def load_dnn_face_detector():
    """
    Load OpenCV DNN-based face detection model
    Downloads model files if they don't exist
    """
    model_dir = os.path.join(DATA_DIR, "face_detection_model")
    os.makedirs(model_dir, exist_ok=True)
    
    prototxt_path = os.path.join(model_dir, "deploy.prototxt")
    model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    
    # Download model files if they don't exist
    if not os.path.exists(prototxt_path):
        print("Downloading face detection prototxt...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            prototxt_path
        )
    
    if not os.path.exists(model_path):
        print("Downloading face detection model (this may take a moment)...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            model_path
        )
    
    return cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Initialize DNN face detector
try:
    face_net = load_dnn_face_detector()
    print("DNN face detector loaded successfully")
except Exception as e:
    print(f"Error loading DNN face detector: {e}")
    # Fallback to Haar cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_net = None

def detect_faces(frame):
    """
    Detect faces using OpenCV DNN or Haar cascades as fallback
    Returns list of (x, y, w, h) tuples
    """
    if face_net is not None:
        return detect_faces_dnn(frame)
    else:
        return detect_faces_haar(frame)

def detect_faces_dnn(frame):
    """
    Detect faces using OpenCV DNN
    """
    h, w = frame.shape[:2]
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
    
    # Set input to the model
    face_net.setInput(blob)
    
    # Run forward pass
    detections = face_net.forward()
    
    faces = []
    confidence_threshold = 0.5
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            # Get bounding box coordinates
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            
            # Convert to (x, y, w, h) format
            x = max(0, x1)
            y = max(0, y1)
            width = min(x2 - x1, w - x)
            height = min(y2 - y1, h - y)
            
            if width > 0 and height > 0:
                faces.append((x, y, width, height))
    
    return faces

def detect_faces_haar(frame):
    """
    Fallback face detection using Haar cascades
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.3, 5)

def preprocess_face(face_img):
    """
    Preprocess face image for embedding extraction
    Always returns the same shape to avoid TensorFlow retracing
    """
    # Ensure face_img is valid
    if face_img is None or face_img.size == 0:
        return None
    
    # Resize to fixed size (224, 224) to ensure consistent input shape
    face_img = cv2.resize(face_img, (224, 224))
    
    # Convert to array and add batch dimension
    arr = img_to_array(face_img)
    arr = np.expand_dims(arr, axis=0)
    
    # Preprocess for ResNet50
    return preprocess_input(arr)

def get_embedding(face_img):
    """
    Extract embedding from face image with optimized TensorFlow calls
    """
    # Preprocess the face
    arr = preprocess_face(face_img)
    if arr is None:
        return None
    
    # Predict with consistent input shape to avoid retracing
    try:
        embedding = model.predict(arr, verbose=0)[0]
        return normalize([embedding])[0]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def get_embeddings_batch(face_images):
    """
    Process multiple faces in a single batch to improve efficiency
    """
    if not face_images:
        return []
    
    # Preprocess all faces
    batch_data = []
    valid_indices = []
    
    for i, face_img in enumerate(face_images):
        arr = preprocess_face(face_img)
        if arr is not None:
            batch_data.append(arr[0])  # Remove the batch dimension for stacking
            valid_indices.append(i)
    
    if not batch_data:
        return []
    
    # Stack into a single batch
    batch_array = np.stack(batch_data, axis=0)
    
    try:
        # Single prediction call for all faces
        embeddings = model.predict(batch_array, verbose=0)
        normalized_embeddings = normalize(embeddings)
        
        # Return embeddings in original order
        result = [None] * len(face_images)
        for i, embedding in zip(valid_indices, normalized_embeddings):
            result[i] = embedding
        
        return result
    except Exception as e:
        print(f"Error getting batch embeddings: {e}")
        return [None] * len(face_images)

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
    
    if faces:
        # Extract all face images
        face_images = []
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size > 0:  # Check if face is valid
                face_images.append(face)
        
        # Process faces in batch for efficiency
        if face_images:
            if len(face_images) == 1:
                # Single face - use regular method
                embedding = get_embedding(face_images[0])
                if embedding is not None:
                    student_id = enrollment_data.get('student_id')
                    if student_id:
                        if 'embeddings' not in enrollment_data:
                            enrollment_data['embeddings'] = []
                        enrollment_data['embeddings'].append(embedding)
                        result['samples'] = len(enrollment_data['embeddings'])
            else:
                # Multiple faces - use batch processing
                embeddings = get_embeddings_batch(face_images)
                student_id = enrollment_data.get('student_id')
                if student_id:
                    if 'embeddings' not in enrollment_data:
                        enrollment_data['embeddings'] = []
                    
                    # Add valid embeddings
                    for embedding in embeddings:
                        if embedding is not None:
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
    
    if faces:
        # Extract all face images
        face_images = []
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size > 0:  # Check if face is valid
                face_images.append(face)
        
        # Process faces efficiently
        if face_images:
            if len(face_images) == 1:
                # Single face
                emb = get_embedding(face_images[0])
                if emb is not None:
                    recognized_faces.extend(recognize_face(emb, embeddings))
            else:
                # Multiple faces - batch processing
                embs = get_embeddings_batch(face_images)
                for emb in embs:
                    if emb is not None:
                        recognized_faces.extend(recognize_face(emb, embeddings))
    
    return {'faces': len(faces), 'recognized': recognized_faces}

def recognize_face(embedding, stored_embeddings, threshold=0.6):
    """
    Recognize a single face embedding against stored embeddings
    Returns list of recognized student IDs
    """
    best_match, best_dist = "Unknown", float("inf")
    
    for sid, stored_emb in stored_embeddings.items():
        try:
            dist = np.linalg.norm(stored_emb - embedding)
            if dist < best_dist:
                best_match, best_dist = sid, dist
        except Exception as e:
            print(f"Error comparing embeddings for {sid}: {e}")
            continue
    
    if best_dist < threshold and best_match != "Unknown":
        attendance_data.add(best_match)
        return [best_match]
    
    return []

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