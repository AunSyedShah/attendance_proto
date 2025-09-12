import cv2
import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import subprocess
import platform
import time
from datetime import datetime
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced models with better error handling
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
    print("✅ MTCNN face detection available")
except ImportError:
    MTCNN_AVAILABLE = False
    print("⚠️  MTCNN not available. Using Haar cascades (fallback)")

try:
    import torch
    from facenet_pytorch import MTCNN as PyTorch_MTCNN, InceptionResnetV1
    FACENET_AVAILABLE = True
    print("✅ FaceNet model available")
except ImportError:
    FACENET_AVAILABLE = False
    print("⚠️  FaceNet not available. Install PyTorch and facenet-pytorch for better accuracy")

# Alternative face recognition libraries that are easier to install
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ Face Recognition library available")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️  Face Recognition library not available")

# Fallback imports (always available)
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------- CONFIG ----------------
DATA_DIR = "data"
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.pkl")
ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.csv")
CONFIG_FILE = os.path.join(DATA_DIR, "config.pkl")

os.makedirs(DATA_DIR, exist_ok=True)

# Updated model configurations with available models
MODEL_CONFIGS = {
    "FaceNet": {"size": (160, 160), "embedding_dim": 512, "accuracy": "High", "speed": "Fast", "available": FACENET_AVAILABLE},
    "Face-Recognition": {"size": (128, 128), "embedding_dim": 128, "accuracy": "High", "speed": "Medium", "available": FACE_RECOGNITION_AVAILABLE},
    "EfficientNetB0": {"size": (224, 224), "embedding_dim": 1280, "accuracy": "High", "speed": "Fast", "available": True},
    "MobileNetV2": {"size": (224, 224), "embedding_dim": 1280, "accuracy": "Medium", "speed": "Very Fast", "available": True},
    "ResNet50": {"size": (224, 224), "embedding_dim": 2048, "accuracy": "Medium", "speed": "Slow", "available": True}
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

# ---------------- ENHANCED MODEL LOADING ----------------
class FaceRecognitionModel:
    def __init__(self, model_name="EfficientNetB0"):
        self.model_name = model_name
        self.model = None
        self.face_detector = None
        self.config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["EfficientNetB0"])
        
        # Check if model is available
        if not self.config["available"]:
            print(f"⚠️  {model_name} not available, falling back to EfficientNetB0")
            self.model_name = "EfficientNetB0"
            self.config = MODEL_CONFIGS["EfficientNetB0"]
        
        self.load_model()
        self.load_face_detector()
    
    def load_model(self):
        """Load the selected face recognition model"""
        try:
            if self.model_name == "FaceNet" and FACENET_AVAILABLE:
                self.model = InceptionResnetV1(pretrained='vggface2').eval()
                print("✅ Loaded FaceNet model")
            
            elif self.model_name == "Face-Recognition" and FACE_RECOGNITION_AVAILABLE:
                # face_recognition library handles model internally
                self.model = "face_recognition"
                print("✅ Loaded Face-Recognition library")
            
            elif self.model_name == "EfficientNetB0":
                self.model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
                print("✅ Loaded EfficientNetB0 model")
            
            elif self.model_name == "MobileNetV2":
                self.model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
                print("✅ Loaded MobileNetV2 model")
            
            else:
                # Fallback to ResNet50
                self.model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
                self.model_name = "ResNet50"
                self.config = MODEL_CONFIGS["ResNet50"]
                print("✅ Loaded ResNet50 model (fallback)")
                
        except Exception as e:
            print(f"❌ Error loading {self.model_name}: {e}")
            # Ultimate fallback
            self.model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
            self.model_name = "ResNet50"
            self.config = MODEL_CONFIGS["ResNet50"]
            print("✅ Loaded ResNet50 model (ultimate fallback)")
    
    def load_face_detector(self):
        """Load face detection model"""
        try:
            if MTCNN_AVAILABLE:
                self.face_detector = MTCNN(min_face_size=20, thresholds=[0.6, 0.7, 0.8])
                print("✅ Using MTCNN face detector")
            else:
                # Fallback to Haar cascades
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                print("✅ Using Haar cascade face detector")
        except Exception as e:
            print(f"⚠️  Error loading face detector: {e}")
            # Ultimate fallback
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            print("✅ Using Haar cascade face detector (fallback)")
    
    def detect_faces(self, frame):
        """Enhanced face detection with alignment"""
        try:
            if MTCNN_AVAILABLE and isinstance(self.face_detector, MTCNN):
                # MTCNN detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = self.face_detector.detect_faces(rgb_frame)
                
                faces = []
                for detection in detections:
                    if detection['confidence'] > 0.85:  # Lower threshold for better detection
                        x, y, w, h = detection['box']
                        padding = 10
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(frame.shape[1] - x, w + 2*padding)
                        h = min(frame.shape[0] - y, h + 2*padding)
                        faces.append((x, y, w, h))
                return faces
            else:
                # Haar cascade
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                return faces
        except Exception as e:
            print(f"⚠️  Face detection error: {e}")
            return []
    
    def preprocess_face(self, face_img):
        """Enhanced face preprocessing"""
        target_size = self.config["size"]
        
        try:
            # Resize face
            face_img = cv2.resize(face_img, target_size)
            
            # Normalize based on model
            if self.model_name == "FaceNet":
                # Convert to RGB and normalize to [-1, 1]
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img = (face_img - 127.5) / 128.0
                return np.expand_dims(face_img, axis=0).astype(np.float32)
            
            elif self.model_name == "Face-Recognition":
                # Convert to RGB for face_recognition library
                return cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            elif self.model_name == "EfficientNetB0":
                arr = img_to_array(face_img)
                arr = np.expand_dims(arr, axis=0)
                return efficientnet_preprocess(arr)
            
            elif self.model_name == "MobileNetV2":
                arr = img_to_array(face_img)
                arr = np.expand_dims(arr, axis=0)
                return mobilenet_preprocess(arr)
            
            else:  # ResNet50
                arr = img_to_array(face_img)
                arr = np.expand_dims(arr, axis=0)
                return preprocess_input(arr)
                
        except Exception as e:
            print(f"⚠️  Preprocessing error: {e}")
            # Return a default preprocessed image
            face_img = cv2.resize(face_img, (224, 224))
            arr = img_to_array(face_img)
            arr = np.expand_dims(arr, axis=0)
            return preprocess_input(arr)
    
    def get_embedding(self, face_img):
        """Get face embedding with the selected model"""
        start_time = time.time()
        
        try:
            if self.model_name == "Face-Recognition" and FACE_RECOGNITION_AVAILABLE:
                # Use face_recognition library
                import face_recognition
                preprocessed = self.preprocess_face(face_img)
                encodings = face_recognition.face_encodings(preprocessed)
                if encodings:
                    embedding = encodings[0]
                else:
                    embedding = np.zeros(128)  # Default embedding size for face_recognition
            
            elif self.model_name == "FaceNet" and FACENET_AVAILABLE:
                # Use FaceNet
                import torch
                preprocessed = self.preprocess_face(face_img)
                with torch.no_grad():
                    preprocessed_tensor = torch.FloatTensor(preprocessed)
                    embedding = self.model(preprocessed_tensor).numpy()[0]
            
            else:
                # Use TensorFlow/Keras models
                preprocessed = self.preprocess_face(face_img)
                embedding = self.model.predict(preprocessed, verbose=0)[0]
            
            # Normalize embedding
            embedding = normalize([embedding])[0]
            
        except Exception as e:
            print(f"⚠️  Embedding extraction error: {e}")
            # Return zero embedding as fallback
            embedding = np.zeros(self.config["embedding_dim"])
        
        processing_time = time.time() - start_time
        return embedding, processing_time

# Global model instance
face_model = None
monitor = PerformanceMonitor()

# ---------------- CAMERA UTILS ----------------
def get_available_cameras(max_cams=10):
    cameras = []
    system = platform.system()

    if system == "Windows":
        for idx in range(max_cams):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cameras.append((idx, f"Camera {idx}"))
                    cap.release()
            except:
                continue
    else:
        for idx in range(max_cams):
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cameras.append((idx, f"Camera {idx}"))
                    cap.release()
            except:
                continue

    return cameras if cameras else [(0, "Default Camera")]

# ---------------- DATA STORAGE ----------------
def load_embeddings():
    return pickle.load(open(EMBEDDINGS_FILE, "rb")) if os.path.exists(EMBEDDINGS_FILE) else {}

def save_embeddings(embeddings):
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)

def load_config():
    if os.path.exists(CONFIG_FILE):
        return pickle.load(open(CONFIG_FILE, "rb"))
    return {"model": "EfficientNetB0", "threshold": 0.75, "similarity_metric": "cosine"}

def save_config(config):
    with open(CONFIG_FILE, "wb") as f:
        pickle.dump(config, f)

# ---------------- ENHANCED ENROLLMENT ----------------
def enroll_student_enhanced(student_id, cam_index, max_samples=20, min_quality_samples=5):
    """Enhanced enrollment with quality control"""
    embeddings = load_embeddings()
    cap = cv2.VideoCapture(cam_index)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    collected = []
    quality_scores = []
    print(f"Enrolling {student_id}... Press 'q' to stop, 's' to save early")
    
    frame_count = 0
    skip_frames = 3  # Process every 3rd frame
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % skip_frames != 0:
            cv2.imshow("Enhanced Enrollment", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Enhance frame quality
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=5)
        
        faces = face_model.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            
            # Quality checks
            face_area = w * h
            if face_area < 2500:  # Minimum face size
                cv2.putText(frame, "Face too small", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                continue
            
            # Blur detection
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            if blur_score < 80:  # Reduced threshold for easier capture
                cv2.putText(frame, "Face blurry", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                continue
            
            embedding, proc_time = face_model.get_embedding(face)
            collected.append(embedding)
            quality_scores.append(blur_score)
            
            # Visual feedback
            color = (0, 255, 0) if len(collected) >= min_quality_samples else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"Samples: {len(collected)}/{max_samples}", 
                       (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Display stats
        cv2.putText(frame, f"Model: {face_model.model_name}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Samples: {len(collected)}/{max_samples}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Enhanced Enrollment", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(collected) >= max_samples:
            break
        elif key == ord('s') and len(collected) >= min_quality_samples:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(collected) >= min_quality_samples:
        # Calculate weighted mean embedding
        if quality_scores:
            weights = np.array(quality_scores)
            weights = weights / np.sum(weights)
            weighted_embedding = np.average(collected, axis=0, weights=weights)
        else:
            weighted_embedding = np.mean(collected, axis=0)
        
        embeddings[student_id] = {
            'embedding': weighted_embedding,
            'samples': len(collected),
            'avg_quality': np.mean(quality_scores) if quality_scores else 0,
            'model': face_model.model_name
        }
        
        save_embeddings(embeddings)
        messagebox.showinfo("Success", 
                           f"Enrolled {student_id} with {len(collected)} samples\\n"
                           f"Model: {face_model.model_name}")
    else:
        messagebox.showwarning("Failed", 
                              f"Need at least {min_quality_samples} samples.\\n"
                              f"Captured only {len(collected)} samples.")

# ---------------- ENHANCED ATTENDANCE ----------------
def mark_attendance_enhanced(cam_index):
    """Enhanced attendance marking"""
    embeddings = load_embeddings()
    if not embeddings:
        messagebox.showerror("Error", "No enrolled students found!")
        return
    
    config = load_config()
    threshold = config.get("threshold", 0.75)
    similarity_metric = config.get("similarity_metric", "cosine")
    
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    attendance = {}
    recent_detections = {}
    
    print("Enhanced attendance marking... Press 'q' to stop")
    
    frame_count = 0
    skip_frames = 2
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        monitor.update()
        
        if frame_count % skip_frames != 0:
            cv2.imshow("Enhanced Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        start_time = time.time()
        
        faces = face_model.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            
            # Skip small faces
            if w * h < 2500:
                continue
            
            embedding, proc_time = face_model.get_embedding(face)
            
            best_match, best_score = "Unknown", 0
            
            # Compare with stored embeddings
            for student_id, student_data in embeddings.items():
                if isinstance(student_data, dict):
                    stored_embedding = student_data.get('embedding', student_data)
                else:
                    stored_embedding = student_data  # Backward compatibility
                
                if similarity_metric == "cosine":
                    similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
                else:
                    # L2 distance converted to similarity
                    distance = np.linalg.norm(embedding - stored_embedding)
                    similarity = 1 / (1 + distance)
                
                if similarity > best_score:
                    best_match, best_score = student_id, similarity
            
            # Determine match
            is_match = best_score > threshold
            label = best_match if is_match else "Unknown"
            
            # Prevent duplicates
            current_time = time.time()
            if (label != "Unknown" and 
                (label not in recent_detections or 
                 current_time - recent_detections[label] > 5)):
                
                attendance[label] = datetime.now()
                recent_detections[label] = current_time
            
            # Visual feedback
            color = (0, 255, 0) if is_match else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} ({best_score:.2f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if is_match and label in recent_detections:
                cv2.putText(frame, "RECORDED", (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        processing_time = time.time() - start_time
        monitor.update(processing_time)
        
        # Display stats
        stats = monitor.get_stats()
        cv2.putText(frame, f"FPS: {stats['fps']:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Model: {face_model.model_name}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Enhanced Attendance", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save attendance
    if attendance:
        with open(ATTENDANCE_FILE, "a") as f:
            for student_id, timestamp in attendance.items():
                f.write(f"{student_id},{timestamp.strftime('%Y-%m-%d %H:%M:%S')}\\n")
        
        attendance_list = list(attendance.keys())
        stats = monitor.get_stats()
        messagebox.showinfo("Attendance Complete", 
                           f"Recorded: {', '.join(attendance_list)}\\n\\n"
                           f"Performance: {stats['fps']:.1f} FPS")
    else:
        messagebox.showwarning("No Attendance", "No faces recognized")

# ---------------- GUI ----------------
def get_available_models():
    """Get list of available models"""
    available = []
    for model_name, config in MODEL_CONFIGS.items():
        if config["available"]:
            available.append(model_name)
    return available

def initialize_model():
    global face_model
    selected_model = model_var.get()
    
    try:
        face_model = FaceRecognitionModel(selected_model)
        model_info_var.set(f"✅ Loaded: {face_model.model_name}")
        
        # Update threshold based on model
        if face_model.model_name in ["FaceNet", "Face-Recognition"]:
            threshold_var.set(0.8)
        else:
            threshold_var.set(0.75)
            
    except Exception as e:
        messagebox.showerror("Error", f"Failed to initialize model: {e}")
        model_info_var.set("❌ Model loading failed")

def start_enhanced_enrollment():
    if face_model is None:
        messagebox.showerror("Error", "Please initialize model first!")
        return
    
    sid = student_id_var.get().strip()
    if not sid:
        messagebox.showerror("Error", "Enter Student ID")
        return
    
    try:
        cam_index = int(camera_var.get().split(":")[0])
        enroll_student_enhanced(sid, cam_index)
    except Exception as e:
        messagebox.showerror("Error", f"Enrollment failed: {e}")

def start_enhanced_attendance():
    if face_model is None:
        messagebox.showerror("Error", "Please initialize model first!")
        return
    
    try:
        # Save current config
        config = {
            "model": model_var.get(),
            "threshold": threshold_var.get(),
            "similarity_metric": similarity_var.get()
        }
        save_config(config)
        
        cam_index = int(camera_var.get().split(":")[0])
        mark_attendance_enhanced(cam_index)
    except Exception as e:
        messagebox.showerror("Error", f"Attendance marking failed: {e}")

# Create main GUI
root = tk.Tk()
root.title("Windows-Compatible Face Recognition Attendance System")
root.geometry("500x650")

# Model selection frame
model_frame = ttk.LabelFrame(root, text="Model Configuration", padding=10)
model_frame.pack(fill="x", padx=10, pady=5)

available_models = get_available_models()
default_model = "EfficientNetB0" if "EfficientNetB0" in available_models else available_models[0]

ttk.Label(model_frame, text="Available Models:").pack(anchor="w")
model_var = tk.StringVar(value=default_model)
model_combo = ttk.Combobox(model_frame, textvariable=model_var,
                          values=available_models, state="readonly")
model_combo.pack(fill="x", pady=2)

model_info_var = tk.StringVar(value="Click 'Initialize Model' to load")
ttk.Label(model_frame, textvariable=model_info_var, foreground="blue").pack(anchor="w", pady=2)

ttk.Button(model_frame, text="Initialize Model", command=initialize_model).pack(pady=5)

# Camera selection frame
camera_frame = ttk.LabelFrame(root, text="Camera Settings", padding=10)
camera_frame.pack(fill="x", padx=10, pady=5)

cameras = get_available_cameras()
camera_var = tk.StringVar(value=f"{cameras[0][0]}: {cameras[0][1]}")
ttk.Label(camera_frame, text="Select Camera:").pack(anchor="w")
camera_dropdown = ttk.Combobox(camera_frame, textvariable=camera_var,
                              values=[f"{idx}: {name}" for idx, name in cameras],
                              state="readonly")
camera_dropdown.pack(fill="x", pady=2)

# Recognition settings frame
settings_frame = ttk.LabelFrame(root, text="Recognition Settings", padding=10)
settings_frame.pack(fill="x", padx=10, pady=5)

ttk.Label(settings_frame, text="Similarity Threshold:").pack(anchor="w")
threshold_var = tk.DoubleVar(value=0.75)
threshold_scale = ttk.Scale(settings_frame, from_=0.5, to=0.95, 
                           variable=threshold_var, orient="horizontal")
threshold_scale.pack(fill="x", pady=2)
threshold_label = ttk.Label(settings_frame, text="0.75")
threshold_label.pack(anchor="w")

def update_threshold_label(val):
    threshold_label.config(text=f"{float(val):.2f}")

threshold_scale.config(command=update_threshold_label)

ttk.Label(settings_frame, text="Similarity Metric:").pack(anchor="w")
similarity_var = tk.StringVar(value="cosine")
similarity_combo = ttk.Combobox(settings_frame, textvariable=similarity_var,
                                values=["cosine", "euclidean"], state="readonly")
similarity_combo.pack(fill="x", pady=2)

# Enrollment frame
enroll_frame = ttk.LabelFrame(root, text="Student Enrollment", padding=10)
enroll_frame.pack(fill="x", padx=10, pady=5)

student_id_var = tk.StringVar()
ttk.Label(enroll_frame, text="Student ID:").pack(anchor="w")
ttk.Entry(enroll_frame, textvariable=student_id_var).pack(fill="x", pady=2)
ttk.Button(enroll_frame, text="Enroll Student", 
          command=start_enhanced_enrollment).pack(pady=5)

# Attendance frame
attendance_frame = ttk.LabelFrame(root, text="Attendance Marking", padding=10)
attendance_frame.pack(fill="x", padx=10, pady=5)

ttk.Button(attendance_frame, text="Mark Attendance", 
          command=start_enhanced_attendance).pack(pady=5)

# Info frame
info_frame = ttk.LabelFrame(root, text="Windows-Compatible System", padding=10)
info_frame.pack(fill="x", padx=10, pady=5)

info_text = f"""
✅ Available Models: {', '.join(available_models)}
✅ No Visual Studio Build Tools required
✅ Works with basic Python packages
✅ Automatic fallback system
✅ Enhanced accuracy and speed

Features:
• Quality-based enrollment
• Real-time performance monitoring  
• Configurable similarity metrics
• Duplicate prevention system
"""

ttk.Label(info_frame, text=info_text, justify="left").pack(anchor="w")

# Initialize with default model
try:
    initialize_model()
except Exception as e:
    print(f"Initial model loading failed: {e}")

root.mainloop()