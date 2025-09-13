from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, flash, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField, SubmitField
from wtforms.validators import DataRequired, Length
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import cv2
import os
import pickle
import numpy as np
import subprocess
import platform
import base64
import io
from PIL import Image
from datetime import datetime
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import threading
import time
import faiss
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

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# ---------------- DATABASE MODELS ----------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='student')  # 'student' or 'admin'
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------- FORMS ----------------
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    role = SelectField('Role', choices=[('student', 'Student'), ('admin', 'Admin')], default='student')
    submit = SubmitField('Register')

# ---------------- DECORATORS ----------------
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ---------------- CONFIG ----------------
DATA_DIR = "data"
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.pkl")
ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.csv")
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# Global variables for camera and state management
camera = None
current_mode = None  # 'enrollment' or 'attendance'
enrollment_data = {}
attendance_data = set()

# Session management variables
current_session = None
session_data = {}

# Daily attendance tracking - format: {date: {student_id: {'in_time': datetime, 'out_time': datetime}}}
daily_attendance = {}

# Store last daily attendance result for user feedback
last_daily_result = None

# Attendance file path
DAILY_ATTENDANCE_FILE = os.path.join(os.path.dirname(__file__), 'data', 'daily_attendance.csv')

# FAISS index file path
FAISS_INDEX_FILE = os.path.join(os.path.dirname(__file__), 'data', 'embeddings.faiss')
FAISS_METADATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'embeddings_metadata.pkl')

# ---------------- FAISS EMBEDDING MANAGER ----------------
class FAISSEmbeddingManager:
    def __init__(self, embedding_dim=2048):
        self.embedding_dim = embedding_dim
        self.index = None
        self.student_ids = []  # Map index position to student ID
        self.load_index()
    
    def load_index(self):
        """Load FAISS index and metadata from disk"""
        try:
            if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_METADATA_FILE):
                # Load FAISS index
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                
                # Load student ID mapping
                with open(FAISS_METADATA_FILE, 'rb') as f:
                    self.student_ids = pickle.load(f)
                
                print(f"FAISS index loaded: {len(self.student_ids)} students, dimension {self.embedding_dim}")
            else:
                # Create new index
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.student_ids = []
                print("Created new FAISS index")
                
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            # Fallback to new index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.student_ids = []
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            
            # Save student ID mapping
            with open(FAISS_METADATA_FILE, 'wb') as f:
                pickle.dump(self.student_ids, f)
                
            print(f"FAISS index saved: {len(self.student_ids)} students")
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
    
    def add_embedding(self, student_id, embedding):
        """Add or update embedding for a student"""
        try:
            # Ensure embedding is the right shape and type
            embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
            
            # Check if student already exists
            if student_id in self.student_ids:
                # Remove old embedding
                old_index = self.student_ids.index(student_id)
                self.remove_embedding(student_id)
            
            # Add new embedding
            self.index.add(embedding)
            self.student_ids.append(student_id)
            
            print(f"Added embedding for {student_id} (total: {len(self.student_ids)})")
            self.save_index()
            
        except Exception as e:
            print(f"Error adding embedding for {student_id}: {e}")
    
    def remove_embedding(self, student_id):
        """Remove embedding for a student (rebuild index without this student)"""
        try:
            if student_id not in self.student_ids:
                return
            
            # Get all embeddings except the one to remove
            all_embeddings = []
            remaining_ids = []
            
            for i, sid in enumerate(self.student_ids):
                if sid != student_id:
                    # Get embedding from index
                    embedding = self.index.reconstruct(i)
                    all_embeddings.append(embedding)
                    remaining_ids.append(sid)
            
            # Rebuild index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            if all_embeddings:
                embeddings_array = np.array(all_embeddings, dtype=np.float32)
                self.index.add(embeddings_array)
            
            self.student_ids = remaining_ids
            self.save_index()
            
            print(f"Removed embedding for {student_id}")
            
        except Exception as e:
            print(f"Error removing embedding for {student_id}: {e}")
    
    def search(self, query_embedding, k=1, threshold=0.6):
        """Search for similar embeddings"""
        try:
            if self.index.ntotal == 0:
                return [], []
            
            # Ensure query embedding is the right shape and type
            query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            
            # Search in FAISS index
            distances, indices = self.index.search(query_embedding, min(k, len(self.student_ids)))
            
            # Convert to lists and filter by threshold
            results = []
            result_distances = []
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and dist < threshold and idx < len(self.student_ids):
                    results.append(self.student_ids[idx])
                    result_distances.append(dist)
            
            return results, result_distances
            
        except Exception as e:
            print(f"Error searching embeddings: {e}")
            return [], []
    
    def get_all_embeddings(self):
        """Get all embeddings as a dictionary (for compatibility)"""
        try:
            embeddings_dict = {}
            for i, student_id in enumerate(self.student_ids):
                embedding = self.index.reconstruct(i)
                embeddings_dict[student_id] = embedding
            return embeddings_dict
        except Exception as e:
            print(f"Error getting all embeddings: {e}")
            return {}
    
    def migrate_from_pickle(self, pickle_file):
        """Migrate embeddings from old pickle format to FAISS"""
        try:
            if os.path.exists(pickle_file):
                with open(pickle_file, 'rb') as f:
                    old_embeddings = pickle.load(f)
                
                print(f"Migrating {len(old_embeddings)} embeddings from pickle to FAISS...")
                
                # Clear current index
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.student_ids = []
                
                # Add all embeddings
                for student_id, embedding in old_embeddings.items():
                    self.add_embedding(student_id, embedding)
                
                print("Migration completed successfully")
                return True
        except Exception as e:
            print(f"Error migrating from pickle: {e}")
        return False

# Initialize FAISS manager
faiss_manager = FAISSEmbeddingManager()

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
    """Load embeddings using FAISS manager"""
    global faiss_manager
    
    # Check if old pickle file exists and migrate
    if os.path.exists(EMBEDDINGS_FILE) and not os.path.exists(FAISS_INDEX_FILE):
        print("Migrating embeddings from pickle to FAISS...")
        faiss_manager.migrate_from_pickle(EMBEDDINGS_FILE)
        # Backup old file
        backup_file = EMBEDDINGS_FILE + ".backup"
        os.rename(EMBEDDINGS_FILE, backup_file)
        print(f"Old embeddings backed up to {backup_file}")
    
    # Return embeddings as dictionary for compatibility
    return faiss_manager.get_all_embeddings()

def save_embeddings(embeddings):
    """Save embeddings using FAISS manager"""
    global faiss_manager
    
    # Clear current index and add all embeddings
    faiss_manager.index = faiss.IndexFlatL2(faiss_manager.embedding_dim)
    faiss_manager.student_ids = []
    
    for student_id, embedding in embeddings.items():
        faiss_manager.add_embedding(student_id, embedding)

# ---------------- SESSION MANAGEMENT ----------------
def create_attendance_session(session_name="Attendance Session"):
    """Create a new attendance session"""
    global current_session, session_data
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_session = {
        'session_id': session_id,
        'name': session_name,
        'start_time': datetime.now(),
        'end_time': None,
        'attendees': {},  # {student_id: {'in_time': datetime, 'out_time': datetime}}
        'absentees': set(),
        'status': 'active'
    }
    
    session_data[session_id] = current_session
    return session_id

def end_attendance_session():
    """End the current attendance session"""
    global current_session
    
    if current_session:
        current_session['end_time'] = datetime.now()
        current_session['status'] = 'completed'
        
        # Mark absent students
        all_students = set(load_embeddings().keys())
        present_students = set(current_session['attendees'].keys())
        absent_students = all_students - present_students
        current_session['absentees'] = absent_students
        
        # Save session to file
        save_session_to_file(current_session)
        
        return current_session['session_id']
    return None

def mark_student_attendance(student_id, action='in'):
    """Mark student in/out time"""
    global current_session
    
    if not current_session:
        return False
    
    if student_id not in current_session['attendees']:
        current_session['attendees'][student_id] = {'in_time': None, 'out_time': None}
    
    if action == 'in' and not current_session['attendees'][student_id]['in_time']:
        current_session['attendees'][student_id]['in_time'] = datetime.now()
    elif action == 'out':
        current_session['attendees'][student_id]['out_time'] = datetime.now()
    
    return True

def save_session_to_file(session):
    """Save session data to CSV file"""
    file_exists = os.path.exists(SESSIONS_FILE)
    
    with open(SESSIONS_FILE, "a", newline='') as f:
        if not file_exists:
            f.write("session_id,session_name,start_time,end_time,student_id,in_time,out_time,status\n")
        
        # Write attendees
        for student_id, times in session['attendees'].items():
            in_time = times['in_time'].strftime('%Y-%m-%d %H:%M:%S') if times['in_time'] else ''
            out_time = times['out_time'].strftime('%Y-%m-%d %H:%M:%S') if times['out_time'] else ''
            
            f.write(f"{session['session_id']},{session['name']},{session['start_time'].strftime('%Y-%m-%d %H:%M:%S')},"
                   f"{session['end_time'].strftime('%Y-%m-%d %H:%M:%S') if session['end_time'] else ''},"
                   f"{student_id},{in_time},{out_time},present\n")
        
        # Write absentees
        for student_id in session['absentees']:
            f.write(f"{session['session_id']},{session['name']},{session['start_time'].strftime('%Y-%m-%d %H:%M:%S')},"
                   f"{session['end_time'].strftime('%Y-%m-%d %H:%M:%S') if session['end_time'] else ''},"
                   f"{student_id},,,absent\n")

# ---------------- DAILY ATTENDANCE SYSTEM ----------------
def load_daily_attendance():
    """Load daily attendance data from CSV file"""
    global daily_attendance
    
    if not os.path.exists(DAILY_ATTENDANCE_FILE):
        # Create the data directory if it doesn't exist
        os.makedirs(os.path.dirname(DAILY_ATTENDANCE_FILE), exist_ok=True)
        return {}
    
    daily_attendance = {}
    try:
        with open(DAILY_ATTENDANCE_FILE, "r") as f:
            lines = f.readlines()
            if len(lines) <= 1:  # Only header or empty
                return {}
            
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    date, student_id, in_time_str, out_time_str = parts[:4]
                    
                    if date not in daily_attendance:
                        daily_attendance[date] = {}
                    
                    daily_attendance[date][student_id] = {
                        'in_time': datetime.strptime(in_time_str, '%Y-%m-%d %H:%M:%S') if in_time_str else None,
                        'out_time': datetime.strptime(out_time_str, '%Y-%m-%d %H:%M:%S') if out_time_str else None
                    }
    except Exception as e:
        print(f"Error loading daily attendance: {e}")
        daily_attendance = {}
    
    return daily_attendance

def save_daily_attendance():
    """Save daily attendance data to CSV file"""
    global daily_attendance
    
    try:
        # Create the data directory if it doesn't exist
        os.makedirs(os.path.dirname(DAILY_ATTENDANCE_FILE), exist_ok=True)
        
        with open(DAILY_ATTENDANCE_FILE, "w") as f:
            f.write("date,student_id,in_time,out_time\n")
            
            for date, students in daily_attendance.items():
                for student_id, times in students.items():
                    in_time_str = times['in_time'].strftime('%Y-%m-%d %H:%M:%S') if times['in_time'] else ''
                    out_time_str = times['out_time'].strftime('%Y-%m-%d %H:%M:%S') if times['out_time'] else ''
                    f.write(f"{date},{student_id},{in_time_str},{out_time_str}\n")
    except Exception as e:
        print(f"Error saving daily attendance: {e}")

def mark_daily_attendance(student_id):
    """
    Mark attendance for a student with automatic in/out detection and 1-minute threshold
    Returns: dict with status info
    """
    global daily_attendance
    
    current_time = datetime.now()
    today = current_time.strftime('%Y-%m-%d')
    
    # Initialize today's attendance if not exists
    if today not in daily_attendance:
        daily_attendance[today] = {}
    
    # Initialize student record for today if not exists
    if student_id not in daily_attendance[today]:
        daily_attendance[today][student_id] = {'in_time': None, 'out_time': None}
    
    student_today = daily_attendance[today][student_id]
    
    # Check if student has no record for today - mark as IN
    if student_today['in_time'] is None:
        student_today['in_time'] = current_time
        save_daily_attendance()
        
        return {
            'action': 'in',
            'time': current_time.strftime('%H:%M:%S'),
            'message': f'Time IN marked for {student_id}',
            'student_id': student_id
        }
    
    # Student has in_time, check if they can mark out (1-minute threshold)
    elif student_today['out_time'] is None:
        time_diff = (current_time - student_today['in_time']).total_seconds()
        
        if time_diff < 60:  # Less than 1 minute
            return {
                'action': 'rejected',
                'message': f'Please wait {int(60 - time_diff)} seconds before marking time OUT',
                'student_id': student_id,
                'remaining_seconds': int(60 - time_diff)
            }
        
        # Mark as OUT
        student_today['out_time'] = current_time
        save_daily_attendance()
        
        # Calculate total time spent
        total_minutes = int((current_time - student_today['in_time']).total_seconds() / 60)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        
        return {
            'action': 'out',
            'time': current_time.strftime('%H:%M:%S'),
            'message': f'Time OUT marked for {student_id}',
            'student_id': student_id,
            'total_time': f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m",
            'in_time': student_today['in_time'].strftime('%H:%M:%S'),
            'out_time': current_time.strftime('%H:%M:%S')
        }
    
    # Student already has both in and out time for today
    else:
        in_time_str = student_today['in_time'].strftime('%H:%M:%S')
        out_time_str = student_today['out_time'].strftime('%H:%M:%S')
        
        return {
            'action': 'completed',
            'message': f'{student_id} already completed attendance today',
            'student_id': student_id,
            'in_time': in_time_str,
            'out_time': out_time_str
        }

def get_today_attendance_summary():
    """Get summary of today's attendance"""
    global daily_attendance
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    if today not in daily_attendance:
        return {
            'date': today,
            'total_students': 0,
            'present_students': 0,
            'completed_students': 0,
            'students': []
        }
    
    students_data = []
    present_count = 0
    completed_count = 0
    
    for student_id, times in daily_attendance[today].items():
        student_info = {
            'student_id': student_id,
            'in_time': times['in_time'].strftime('%H:%M:%S') if times['in_time'] else None,
            'out_time': times['out_time'].strftime('%H:%M:%S') if times['out_time'] else None,
            'status': 'absent'
        }
        
        if times['in_time']:
            present_count += 1
            student_info['status'] = 'present'
            
            if times['out_time']:
                completed_count += 1
                student_info['status'] = 'completed'
                
                # Calculate total time
                total_seconds = (times['out_time'] - times['in_time']).total_seconds()
                total_minutes = int(total_seconds / 60)
                hours = total_minutes // 60
                minutes = total_minutes % 60
                student_info['total_time'] = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
        
        students_data.append(student_info)
    
    return {
        'date': today,
        'total_students': len(students_data),
        'present_students': present_count,
        'completed_students': completed_count,
        'students': students_data
    }

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
    
    print(f"Enrollment processing: {len(faces)} faces detected, current samples: {len(enrollment_data.get('embeddings', []))}")
    
    if faces:
        # Extract all face images
        face_images = []
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size > 0:  # Check if face is valid
                face_images.append(face)
        
        print(f"Extracted {len(face_images)} valid face images")
        
        # Process faces in batch for efficiency
        if face_images:
            if len(face_images) == 1:
                # Single face - use regular method
                embedding = get_embedding(face_images[0])
                print(f"Generated embedding: {embedding is not None}")
                if embedding is not None:
                    student_id = enrollment_data.get('student_id')
                    if student_id:
                        if 'embeddings' not in enrollment_data:
                            enrollment_data['embeddings'] = []
                        enrollment_data['embeddings'].append(embedding)
                        result['samples'] = len(enrollment_data['embeddings'])
                        print(f"Added embedding for {student_id}, total samples: {result['samples']}")
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
    Recognize a single face embedding using FAISS vector search
    Returns list of recognized student IDs
    """
    global faiss_manager, last_daily_result
    
    try:
        # Use FAISS for efficient similarity search
        results, distances = faiss_manager.search(embedding, k=1, threshold=threshold)
        
        if results:
            best_match = results[0]
            best_dist = distances[0]
            
            print(f"FAISS recognition: {best_match} with distance {best_dist:.4f}")
            
            attendance_data.add(best_match)
            # Mark attendance using enhanced daily system
            daily_result = mark_daily_attendance(best_match)
            last_daily_result = daily_result  # Store for user feedback
            print(f"Daily attendance result for {best_match}: {daily_result['action']} - {daily_result['message']}")
            
            # Also mark in current session for compatibility
            mark_student_attendance(best_match, 'in')
            return [best_match]
        
        return []
        
    except Exception as e:
        print(f"Error in FAISS recognition: {e}")
        # Fallback to original method if FAISS fails
        return recognize_face_fallback(embedding, stored_embeddings, threshold)

def recognize_face_fallback(embedding, stored_embeddings, threshold=0.6):
    """
    Fallback recognition using numpy distance calculations
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
        global last_daily_result
        attendance_data.add(best_match)
        # Mark attendance using enhanced daily system
        daily_result = mark_daily_attendance(best_match)
        last_daily_result = daily_result  # Store for user feedback
        print(f"Fallback recognition result for {best_match}: {daily_result['action']} - {daily_result['message']}")
        
        # Also mark in current session for compatibility
        mark_student_attendance(best_match, 'in')
        return [best_match]
    
    return []

# ---------------- AUTHENTICATION ROUTES ----------------
@app.route('/')
def index():
    if current_user.is_authenticated and current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    # Default route for students - direct access to attendance
    return redirect(url_for('attendance'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Always logout any existing user to ensure clean slate
    if current_user.is_authenticated:
        logout_user()
        # Clear session data
        session.clear()
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            next_page = request.args.get('next')
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(next_page) if next_page else redirect(url_for('index'))
        flash('Invalid username or password', 'error')
    
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already exists', 'error')
            return render_template('register.html', form=form)
        
        user = User(username=form.username.data, role=form.role.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()  # Clear all session data
    flash('You have been logged out', 'info')
    return redirect(url_for('attendance'))  # Redirect to public attendance page



# ---------------- STUDENT ROUTES ----------------
@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

# ---------------- ADMIN ROUTES ----------------
@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/enrollment')
@login_required
@admin_required
def enrollment():
    return render_template('enrollment.html')

@app.route('/config')
@login_required  
@admin_required
def config():
    # Default configuration values for the template
    config_data = {
        'model': 'ResNet50',
        'threshold': 0.7,
        'similarity_metric': 'cosine',
        'max_faces': 10,
        'frame_skip': 2,
        'enable_gpu': False,
        'log_level': 'INFO'
    }
    
    # Available models list
    models = ['ResNet50', 'VGGFace', 'FaceNet', 'DeepFace']
    
    return render_template('config.html', config=config_data, models=models)

# ---------------- API ROUTES ----------------

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global current_mode
    
    # Handle FormData from attendance page (blob frame) or JSON data
    if 'frame' in request.files:
        # FormData from attendance page
        frame_file = request.files['frame']
        if frame_file:
            # Convert blob to base64
            frame_data = frame_file.read()
            import base64
            image_data = base64.b64encode(frame_data).decode('utf-8')
            mode = 'attendance'
            
            # Auto-set mode for attendance if not set
            if current_mode is None:
                current_mode = 'attendance'
        else:
            return jsonify({'status': 'error', 'message': 'No frame data provided'})
    else:
        # JSON data from other sources  
        data = request.json or {}
        image_data = data.get('image')
        mode = data.get('mode', 'attendance')
        
        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image data provided'})
        
        # Check if camera/processing should be active for enrollment mode
        if mode == 'enrollment' and current_mode is None:
            return jsonify({'status': 'error', 'message': 'Camera not active for enrollment'})
        
        # Auto-set mode for attendance if not set
        if current_mode is None and mode == 'attendance':
            current_mode = 'attendance'
    
    # Validate mode matches current mode (but allow attendance to auto-start)
    if current_mode != mode and not (current_mode is None and mode == 'attendance'):
        return jsonify({'status': 'error', 'message': f'Mode mismatch: expected {current_mode}, got {mode}'})
    
    # Double-check current_mode before processing
    if current_mode is None:
        return jsonify({'status': 'error', 'message': 'Camera processing not available'})
    
    # Convert base64 to image for both enrollment and attendance
    try:
        import base64
        import io
        from PIL import Image
        import numpy as np
        
        # Remove data:image/jpeg;base64, prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        frame = np.array(image)
        
        if current_mode == 'enrollment':
            result = process_enrollment_frame(frame)
            if result is None:
                return jsonify({'status': 'error', 'message': 'Failed to process enrollment frame'})
            return jsonify({
                'status': 'success',
                'faces': result['faces'],
                'samples': result['samples'],
                'message': f"Processed frame: {result['faces']} faces detected, {result['samples']} samples collected"
            })
        else:
            # Process attendance frame
            result = process_attendance_frame(frame)
            
            if result is None:
                return jsonify({'status': 'error', 'message': 'Failed to process attendance frame'})
            
            # Format response for frontend
            if result['faces'] > 0 and result['recognized']:
                global last_daily_result
                person_name = result['recognized'][0]  # Take first recognized face
                
                # Create enhanced response with daily attendance info
                response = {
                    'status': 'success',
                    'person_name': person_name,
                    'message': f'Attendance marked for {person_name}'
                }
                
                # Add daily attendance details if available
                if last_daily_result and last_daily_result['student_id'] == person_name:
                    response['daily_attendance'] = last_daily_result
                    
                    # Create more descriptive message based on action
                    if last_daily_result['action'] == 'in':
                        response['message'] = f'✅ {person_name} - Time IN marked at {last_daily_result["time"]}'
                    elif last_daily_result['action'] == 'out':
                        response['message'] = f'🚪 {person_name} - Time OUT marked at {last_daily_result["time"]} (Total: {last_daily_result.get("total_time", "N/A")})'
                    elif last_daily_result['action'] == 'rejected':
                        response['message'] = f'⏱️ {person_name} - {last_daily_result["message"]}'
                    elif last_daily_result['action'] == 'completed':
                        response['message'] = f'✨ {person_name} - Already completed attendance today'
                
                return jsonify(response)
            elif result['faces'] > 0:
                return jsonify({
                    'status': 'success', 
                    'person_name': 'Unknown',
                    'message': 'Face detected but not recognized'
                })
            else:
                return jsonify({
                    'status': 'info',
                    'message': 'No face detected in frame'
                })
                
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'status': 'error', 'message': 'Error processing image'})

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global current_mode
    
    data = request.json or {}
    mode = data.get('mode', 'attendance')
    current_mode = mode
    
    return jsonify({'status': 'success', 'message': f'Camera started in {mode} mode'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global current_mode, enrollment_data, attendance_data
    
    # Reset all processing states
    current_mode = None
    
    # Clear any incomplete enrollment data
    if enrollment_data:
        print(f"Clearing incomplete enrollment data for: {enrollment_data.get('student_id', 'unknown')}")
        enrollment_data = {}
    
    # Keep attendance_data as it might be intentionally collected
    # but log the current state
    if attendance_data:
        print(f"Current attendance data preserved: {len(attendance_data)} students")
    
    return jsonify({
        'status': 'success', 
        'message': 'Camera stopped gracefully',
        'attendance_count': len(attendance_data)
    })

@app.route('/start_enrollment', methods=['POST'])
@login_required
@admin_required
def start_enrollment():
    global enrollment_data, current_mode
    
    data = request.json or {}
    student_id = data.get('student_id', '').strip()
    
    if not student_id:
        return jsonify({'status': 'error', 'message': 'Student ID is required'})
    
    enrollment_data = {'student_id': student_id, 'embeddings': []}
    current_mode = 'enrollment'  # Set current mode for frame processing
    
    print(f"Started enrollment for {student_id}, current_mode set to: {current_mode}")
    
    return jsonify({'status': 'success', 'message': f'Started enrollment for {student_id}'})

@app.route('/complete_enrollment', methods=['POST'])
@login_required
@admin_required
def complete_enrollment():
    global enrollment_data, current_mode, faiss_manager
    
    if 'student_id' not in enrollment_data or not enrollment_data.get('embeddings'):
        return jsonify({'status': 'error', 'message': 'No enrollment data found'})
    
    student_id = enrollment_data['student_id']
    embeddings_list = enrollment_data['embeddings']
    
    # Calculate mean embedding for better representation
    mean_embedding = np.mean(embeddings_list, axis=0)
    
    # Add to FAISS index directly
    faiss_manager.add_embedding(student_id, mean_embedding)
    
    # Clear enrollment data and mode
    enrollment_data = {}
    current_mode = None
    
    print(f"Completed FAISS enrollment for {student_id} with {len(embeddings_list)} samples")
    
    return jsonify({
        'status': 'success', 
        'message': f'Enrolled {student_id} with {len(embeddings_list)} samples (FAISS)'
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

# ---------------- SESSION MANAGEMENT ROUTES ----------------
@app.route('/start_session', methods=['POST'])
@login_required
@admin_required
def start_session():
    data = request.json or {}
    session_name = data.get('session_name', f'Session_{datetime.now().strftime("%Y%m%d_%H%M")}')
    
    session_id = create_attendance_session(session_name)
    
    return jsonify({
        'status': 'success',
        'message': f'Started attendance session: {session_name}',
        'session_id': session_id
    })

@app.route('/end_session', methods=['POST'])
@login_required
@admin_required
def end_session():
    session_id = end_attendance_session()
    
    if session_id:
        return jsonify({
            'status': 'success',
            'message': f'Ended attendance session: {session_id}',
            'session_id': session_id
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'No active session to end'
        })

@app.route('/current_session', methods=['GET'])
def get_current_session():
    global current_session
    
    # Get today's attendance summary
    daily_summary = get_today_attendance_summary()
    
    if current_session and current_session['status'] == 'active':
        # Return format expected by attendance.html JavaScript
        return jsonify({
            'session_active': True,
            'session_date': current_session['start_time'].strftime('%Y-%m-%d'),
            'session_name': current_session['name'],
            'session_id': current_session['session_id'],
            'total_present': len(current_session['attendees']),
            'last_updated': datetime.now().strftime('%H:%M:%S'),
            'start_time': current_session['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'status': current_session['status'],
            # Enhanced time logging information
            'daily_attendance': daily_summary
        })
    else:
        return jsonify({
            'session_active': False,
            'session_date': None,
            'session_name': None,
            'total_present': 0,
            'last_updated': None,
            # Enhanced time logging information
            'daily_attendance': daily_summary
        })

@app.route('/mark_out', methods=['POST'])
def mark_out():
    data = request.json or {}
    student_id = data.get('student_id', '').strip()
    
    if not student_id:
        return jsonify({'status': 'error', 'message': 'Student ID is required'})
    
    if mark_student_attendance(student_id, 'out'):
        return jsonify({
            'status': 'success',
            'message': f'Marked out time for {student_id}'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'No active session or student not found'
        })

@app.route('/view_attendance')
@login_required
@admin_required
def view_attendance():
    return render_template('attendance_view.html')

# ---------------- DATABASE INITIALIZATION ----------------
def init_db():
    """Initialize database and create default admin user"""
    with app.app_context():
        db.create_all()
        
        # Create default admin user if doesn't exist
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(username='admin', role='admin')
            admin.set_password('admin123')  # Change this in production!
            db.session.add(admin)
            db.session.commit()
            print("Default admin user created: username='admin', password='admin123'")
        
        # Create default student user if doesn't exist
        student = User.query.filter_by(username='student').first()
        if not student:
            student = User(username='student', role='student')
            student.set_password('student123')  # Change this in production!
            db.session.add(student)
            db.session.commit()
            print("Default student user created: username='student', password='student123'")

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)