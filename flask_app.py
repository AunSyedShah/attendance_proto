from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import cv2
import os
import pickle
import numpy as np
import base64
import io
from PIL import Image
import subprocess
import platform
import time
from datetime import datetime
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced models
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("MTCNN not available. Install with: pip install mtcnn")

try:
    from facenet_pytorch import MTCNN as PyTorch_MTCNN, InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    print("FaceNet-PyTorch not available. Install with: pip install facenet-pytorch")

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("InsightFace not available. Install with: pip install insightface")

# Fallback imports
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# ---------------- CONFIG ----------------
DATA_DIR = "data"
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.pkl")
ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.csv")
CONFIG_FILE = os.path.join(DATA_DIR, "config.pkl")

os.makedirs(DATA_DIR, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    "FaceNet": {"size": (160, 160), "embedding_dim": 512, "accuracy": "High", "speed": "Fast"},
    "ArcFace": {"size": (112, 112), "embedding_dim": 512, "accuracy": "Very High", "speed": "Fast"},
    "ResNet50": {"size": (224, 224), "embedding_dim": 2048, "accuracy": "Medium", "speed": "Slow"},
    "MobileNetV2": {"size": (224, 224), "embedding_dim": 1280, "accuracy": "Medium", "speed": "Very Fast"}
}

class PerformanceMonitor:
    def __init__(self):
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_times = []
    
    def update(self, processing_time=None):
        self.frame_count += 1
        current_time = time.time()
        
        if processing_time:
            self.processing_times.append(processing_time)
        
        # Calculate FPS every second
        if current_time - self.start_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.start_time)
            self.frame_count = 0
            self.start_time = current_time
    
    def get_stats(self):
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        return {
            "fps": self.fps,
            "avg_processing_time": avg_processing_time,
            "total_frames": len(self.processing_times)
        }

class FaceRecognitionModel:
    def __init__(self, model_name="MobileNetV2"):
        self.model_name = model_name
        self.model = None
        self.face_detector = None
        self.config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["MobileNetV2"])
        self.load_model()
        self.load_face_detector()
    
    def load_model(self):
        """Load the selected face recognition model"""
        if self.model_name == "FaceNet" and FACENET_AVAILABLE:
            self.model = InceptionResnetV1(pretrained='vggface2').eval()
            print("Loaded FaceNet model")
        
        elif self.model_name == "ArcFace" and INSIGHTFACE_AVAILABLE:
            self.model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            print("Loaded ArcFace model")
        
        elif self.model_name == "MobileNetV2":
            self.model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
            print("Loaded MobileNetV2 model")
        
        else:
            # Fallback to ResNet50
            self.model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
            self.model_name = "ResNet50"
            self.config = MODEL_CONFIGS["ResNet50"]
            print("Loaded ResNet50 model (fallback)")
    
    def load_face_detector(self):
        """Load face detection model"""
        if MTCNN_AVAILABLE:
            self.face_detector = MTCNN(min_face_size=20, thresholds=[0.6, 0.7, 0.8])
            print("Using MTCNN face detector")
        else:
            # Fallback to Haar cascades
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            print("Using Haar cascade face detector (fallback)")
    
    def detect_faces(self, frame):
        """Enhanced face detection with alignment"""
        if MTCNN_AVAILABLE and isinstance(self.face_detector, MTCNN):
            # MTCNN detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.face_detector.detect_faces(rgb_frame)
            
            faces = []
            for detection in detections:
                if detection['confidence'] > 0.9:  # High confidence threshold
                    x, y, w, h = detection['box']
                    # Add some padding
                    padding = 10
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(frame.shape[1] - x, w + 2*padding)
                    h = min(frame.shape[0] - y, h + 2*padding)
                    faces.append((x, y, w, h))
            return faces
        else:
            # Haar cascade fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            return faces
    
    def preprocess_face(self, face_img):
        """Enhanced face preprocessing"""
        target_size = self.config["size"]
        
        # Resize face
        face_img = cv2.resize(face_img, target_size)
        
        # Normalize based on model
        if self.model_name == "FaceNet":
            # Convert to RGB and normalize to [-1, 1]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = (face_img - 127.5) / 128.0
            return np.expand_dims(face_img, axis=0).astype(np.float32)
        
        elif self.model_name == "ArcFace":
            # InsightFace handles preprocessing internally
            return cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        elif self.model_name == "MobileNetV2":
            arr = img_to_array(face_img)
            arr = np.expand_dims(arr, axis=0)
            return mobilenet_preprocess(arr)
        
        else:  # ResNet50
            arr = img_to_array(face_img)
            arr = np.expand_dims(arr, axis=0)
            return preprocess_input(arr)
    
    def get_embedding(self, face_img):
        """Get face embedding with the selected model"""
        start_time = time.time()
        
        if self.model_name == "ArcFace" and INSIGHTFACE_AVAILABLE:
            # Use InsightFace
            preprocessed = self.preprocess_face(face_img)
            faces = self.model.get(preprocessed)
            if faces:
                embedding = faces[0].embedding
                embedding = normalize([embedding])[0]
            else:
                embedding = np.zeros(512)  # Return zero embedding if no face detected
        
        else:
            # Use other models
            preprocessed = self.preprocess_face(face_img)
            
            if self.model_name == "FaceNet":
                import torch
                with torch.no_grad():
                    preprocessed_tensor = torch.FloatTensor(preprocessed)
                    embedding = self.model(preprocessed_tensor).numpy()[0]
            else:
                embedding = self.model.predict(preprocessed, verbose=0)[0]
            
            embedding = normalize([embedding])[0]
        
        processing_time = time.time() - start_time
        return embedding, processing_time

# Global instances
face_model = None
monitor = PerformanceMonitor()

# ---------------- UTILITY FUNCTIONS ----------------
def load_embeddings():
    return pickle.load(open(EMBEDDINGS_FILE, "rb")) if os.path.exists(EMBEDDINGS_FILE) else {}

def save_embeddings(embeddings):
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)

def load_config():
    if os.path.exists(CONFIG_FILE):
        return pickle.load(open(CONFIG_FILE, "rb"))
    return {"model": "MobileNetV2", "threshold": 0.7, "similarity_metric": "cosine"}

def save_config(config):
    with open(CONFIG_FILE, "wb") as f:
        pickle.dump(config, f)

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    # Remove data:image/jpeg;base64, prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    image_data = base64.b64decode(base64_string)
    
    # Convert to PIL Image
    pil_image = Image.open(io.BytesIO(image_data))
    
    # Convert to OpenCV BGR format
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return opencv_image

# ---------------- FLASK ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enrollment')
def enrollment():
    models = list(MODEL_CONFIGS.keys())
    return render_template('enrollment.html', models=models)

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

@app.route('/config')
def config():
    current_config = load_config()
    models = list(MODEL_CONFIGS.keys())
    return render_template('config.html', config=current_config, models=models, model_configs=MODEL_CONFIGS)

@app.route('/api/initialize_model', methods=['POST'])
def initialize_model():
    global face_model
    data = request.get_json()
    model_name = data.get('model', 'MobileNetV2')
    
    try:
        face_model = FaceRecognitionModel(model_name)
        return jsonify({
            'success': True,
            'message': f'Loaded {face_model.model_name} model',
            'model_info': face_model.config
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error loading model: {str(e)}'
        })

@app.route('/api/enroll', methods=['POST'])
def api_enroll():
    global face_model
    
    if face_model is None:
        return jsonify({'success': False, 'message': 'Model not initialized'})
    
    data = request.get_json()
    student_id = data.get('student_id')
    image_data = data.get('image')
    
    if not student_id or not image_data:
        return jsonify({'success': False, 'message': 'Missing student ID or image data'})
    
    try:
        # Convert base64 to image
        frame = base64_to_image(image_data)
        
        # Detect faces
        faces = face_model.detect_faces(frame)
        
        if not faces:
            return jsonify({'success': False, 'message': 'No face detected'})
        
        # Get the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Quality checks
        face_area = w * h
        if face_area < 2500:
            return jsonify({'success': False, 'message': 'Face too small'})
        
        face = frame[y:y+h, x:x+w]
        
        # Blur detection
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        if blur_score < 100:
            return jsonify({'success': False, 'message': 'Face too blurry'})
        
        # Get embedding
        embedding, proc_time = face_model.get_embedding(face)
        
        # Load existing embeddings
        embeddings = load_embeddings()
        
        # Store or update embedding
        if student_id in embeddings:
            # Update existing embedding (weighted average)
            existing_data = embeddings[student_id]
            existing_embedding = existing_data.get('embedding', existing_data)
            existing_samples = existing_data.get('samples', 1) if isinstance(existing_data, dict) else 1
            
            # Weighted average
            total_samples = existing_samples + 1
            new_embedding = (existing_embedding * existing_samples + embedding) / total_samples
            
            embeddings[student_id] = {
                'embedding': new_embedding,
                'samples': total_samples,
                'last_updated': datetime.now().isoformat(),
                'model': face_model.model_name
            }
        else:
            embeddings[student_id] = {
                'embedding': embedding,
                'samples': 1,
                'created': datetime.now().isoformat(),
                'model': face_model.model_name
            }
        
        save_embeddings(embeddings)
        
        return jsonify({
            'success': True,
            'message': f'Successfully enrolled {student_id}',
            'quality_score': float(blur_score),
            'face_area': int(face_area),
            'processing_time': float(proc_time)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing enrollment: {str(e)}'})

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    global face_model, monitor
    
    if face_model is None:
        return jsonify({'success': False, 'message': 'Model not initialized'})
    
    data = request.get_json()
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'message': 'Missing image data'})
    
    try:
        config = load_config()
        threshold = config.get('threshold', 0.7)
        similarity_metric = config.get('similarity_metric', 'cosine')
        
        # Convert base64 to image
        frame = base64_to_image(image_data)
        
        start_time = time.time()
        
        # Detect faces
        faces = face_model.detect_faces(frame)
        
        results = []
        embeddings = load_embeddings()
        
        if not embeddings:
            return jsonify({'success': False, 'message': 'No enrolled students found'})
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            
            # Skip small faces
            if w * h < 2500:
                continue
            
            embedding, proc_time = face_model.get_embedding(face)
            
            best_match, best_score = "Unknown", 0
            
            # Compare with stored embeddings
            for student_id, student_data in embeddings.items():
                stored_embedding = student_data.get('embedding', student_data)
                
                if similarity_metric == "cosine":
                    similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
                else:
                    # L2 distance (convert to similarity)
                    distance = np.linalg.norm(embedding - stored_embedding)
                    similarity = 1 / (1 + distance)
                
                if similarity > best_score:
                    best_match, best_score = student_id, similarity
            
            # Determine if it's a match
            is_match = best_score > threshold
            label = best_match if is_match else "Unknown"
            
            results.append({
                'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                'label': label,
                'confidence': float(best_score),
                'is_match': is_match
            })
        
        processing_time = time.time() - start_time
        monitor.update(processing_time)
        
        stats = monitor.get_stats()
        
        return jsonify({
            'success': True,
            'faces': results,
            'stats': {
                'processing_time': processing_time,
                'fps': stats['fps'],
                'model': face_model.model_name,
                'threshold': threshold
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing recognition: {str(e)}'})

@app.route('/api/mark_attendance', methods=['POST'])
def api_mark_attendance():
    data = request.get_json()
    student_ids = data.get('student_ids', [])
    
    if not student_ids:
        return jsonify({'success': False, 'message': 'No students to mark'})
    
    try:
        timestamp = datetime.now()
        with open(ATTENDANCE_FILE, "a") as f:
            for student_id in student_ids:
                f.write(f"{student_id},{timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return jsonify({
            'success': True,
            'message': f'Marked attendance for {len(student_ids)} students',
            'students': student_ids,
            'timestamp': timestamp.isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error marking attendance: {str(e)}'})

@app.route('/api/save_config', methods=['POST'])
def api_save_config():
    data = request.get_json()
    config = {
        'model': data.get('model', 'MobileNetV2'),
        'threshold': data.get('threshold', 0.7),
        'similarity_metric': data.get('similarity_metric', 'cosine')
    }
    
    try:
        save_config(config)
        return jsonify({'success': True, 'message': 'Configuration saved'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error saving config: {str(e)}'})

@app.route('/api/students')
def api_students():
    try:
        embeddings = load_embeddings()
        students = []
        
        for student_id, data in embeddings.items():
            if isinstance(data, dict):
                students.append({
                    'id': student_id,
                    'samples': data.get('samples', 1),
                    'model': data.get('model', 'Unknown'),
                    'created': data.get('created', 'Unknown'),
                    'last_updated': data.get('last_updated', 'Unknown')
                })
            else:
                students.append({
                    'id': student_id,
                    'samples': 1,
                    'model': 'Unknown',
                    'created': 'Unknown',
                    'last_updated': 'Unknown'
                })
        
        return jsonify({'success': True, 'students': students})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error loading students: {str(e)}'})

if __name__ == '__main__':
    # Initialize default model
    face_model = FaceRecognitionModel("MobileNetV2")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)