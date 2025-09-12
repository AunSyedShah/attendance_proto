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

# ---------------- ENHANCED MODEL LOADING ----------------
class FaceRecognitionModel:
    def __init__(self, model_name="FaceNet"):
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

# Global model instance
face_model = None
monitor = PerformanceMonitor()

# ---------------- CAMERA UTILS ----------------
def get_available_cameras(max_cams=10):
    cameras = []
    system = platform.system()

    if system == "Linux":
        try:
            output = subprocess.check_output("v4l2-ctl --list-devices", shell=True).decode()
            blocks = output.strip().split("\n\n")
            for idx, block in enumerate(blocks):
                lines = block.strip().splitlines()
                if len(lines) >= 2:
                    name = lines[0]
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        cameras.append((idx, name))
                        cap.release()
        except Exception:
            pass
    elif system == "Windows":
        for idx in range(max_cams):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                cameras.append((idx, f"Camera {idx}"))
                cap.release()
    else:
        for idx in range(max_cams):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cameras.append((idx, f"Camera {idx}"))
                cap.release()

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
    return {"model": "MobileNetV2", "threshold": 0.7, "similarity_metric": "cosine"}

def save_config(config):
    with open(CONFIG_FILE, "wb") as f:
        pickle.dump(config, f)

# ---------------- ENHANCED ENROLLMENT ----------------
def enroll_student_enhanced(student_id, cam_index, max_samples=30, min_quality_samples=5):
    """Enhanced enrollment with quality control and data augmentation"""
    embeddings = load_embeddings()
    cap = cv2.VideoCapture(cam_index)
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    collected = []
    quality_scores = []
    print(f"Enrolling {student_id}... Press 'q' to stop, 's' to save early")
    
    frame_count = 0
    skip_frames = 3  # Process every 3rd frame for speed
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue
        
        # Apply histogram equalization for better lighting
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
        
        faces = face_model.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            
            # Quality check: face size
            face_area = w * h
            if face_area < 2500:  # Minimum face size
                cv2.putText(frame, "Face too small", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                continue
            
            # Quality check: blur detection
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            if blur_score < 100:  # Too blurry
                cv2.putText(frame, "Face too blurry", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                continue
            
            embedding, proc_time = face_model.get_embedding(face)
            collected.append(embedding)
            quality_scores.append(blur_score)
            
            # Visual feedback
            color = (0, 255, 0) if len(collected) >= min_quality_samples else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"Samples: {len(collected)}/{max_samples}", 
                       (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Quality: {blur_score:.0f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
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
        # Use weighted average based on quality scores
        weights = np.array(quality_scores)
        weights = weights / np.sum(weights)
        
        # Calculate weighted mean embedding
        weighted_embedding = np.average(collected, axis=0, weights=weights)
        
        embeddings[student_id] = {
            'embedding': weighted_embedding,
            'samples': len(collected),
            'avg_quality': np.mean(quality_scores),
            'model': face_model.model_name
        }
        
        save_embeddings(embeddings)
        messagebox.showinfo("Success", 
                           f"Enrolled {student_id} with {len(collected)} samples\\n"
                           f"Average quality: {np.mean(quality_scores):.1f}\\n"
                           f"Model: {face_model.model_name}")
    else:
        messagebox.showwarning("Failed", 
                              f"Need at least {min_quality_samples} good quality samples.\\n"
                              f"Captured only {len(collected)} samples.")

# ---------------- ENHANCED ATTENDANCE ----------------
def mark_attendance_enhanced(cam_index):
    """Enhanced attendance marking with better accuracy and speed"""
    embeddings = load_embeddings()
    if not embeddings:
        messagebox.showerror("Error", "No enrolled students found!")
        return
    
    config = load_config()
    threshold = config.get("threshold", 0.7)
    similarity_metric = config.get("similarity_metric", "cosine")
    
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    attendance = {}  # student_id -> timestamp
    recent_detections = {}  # Prevent duplicate entries
    
    print("Enhanced attendance marking... Press 'q' to stop")
    
    frame_count = 0
    skip_frames = 2  # Process every 2nd frame for better speed
    
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
            
            # Skip if face is too small
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
            
            # Prevent duplicate entries (5-second cooldown)
            current_time = time.time()
            if (label != "Unknown" and 
                (label not in recent_detections or 
                 current_time - recent_detections[label] > 5)):
                
                attendance[label] = datetime.now()
                recent_detections[label] = current_time
            
            # Visual feedback
            color = (0, 255, 0) if is_match else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display info
            cv2.putText(frame, f"{label} ({best_score:.2f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if is_match and label in recent_detections:
                cv2.putText(frame, "RECORDED", (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        processing_time = time.time() - start_time
        monitor.update(processing_time)
        
        # Display performance stats
        stats = monitor.get_stats()
        cv2.putText(frame, f"FPS: {stats['fps']:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Model: {face_model.model_name}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {threshold:.2f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
        messagebox.showinfo("Attendance Complete", 
                           f"Recorded attendance for:\\n{', '.join(attendance_list)}\\n\\n"
                           f"Performance Stats:\\n"
                           f"Average FPS: {stats['fps']:.1f}\\n"
                           f"Processing time: {stats['avg_processing_time']*1000:.1f}ms")
    else:
        messagebox.showwarning("No Attendance", "No faces recognized")

# ---------------- ENHANCED GUI ----------------
def initialize_model():
    global face_model
    selected_model = model_var.get()
    face_model = FaceRecognitionModel(selected_model)
    
    # Update threshold based on model
    if selected_model in ["FaceNet", "ArcFace"]:
        threshold_var.set(0.8)
    else:
        threshold_var.set(0.7)
    
    model_info_var.set(f"Loaded: {face_model.model_name}")

def start_enhanced_enrollment():
    if face_model is None:
        messagebox.showerror("Error", "Please initialize model first!")
        return
    
    sid = student_id_var.get().strip()
    if not sid:
        messagebox.showerror("Error", "Enter Student ID")
        return
    
    cam_index = int(camera_var.get().split(":")[0])
    enroll_student_enhanced(sid, cam_index)

def start_enhanced_attendance():
    if face_model is None:
        messagebox.showerror("Error", "Please initialize model first!")
        return
    
    # Save current config
    config = {
        "model": model_var.get(),
        "threshold": threshold_var.get(),
        "similarity_metric": similarity_var.get()
    }
    save_config(config)
    
    cam_index = int(camera_var.get().split(":")[0])
    mark_attendance_enhanced(cam_index)

def show_model_comparison():
    """Show model comparison window"""
    comparison_window = tk.Toplevel(root)
    comparison_window.title("Model Comparison")
    comparison_window.geometry("600x400")
    
    # Create comparison table
    columns = ('Model', 'Accuracy', 'Speed', 'Embedding Size', 'Requirements')
    tree = ttk.Treeview(comparison_window, columns=columns, show='headings')
    
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)
    
    # Add model data
    models_data = [
        ('FaceNet', 'High', 'Fast', '512D', 'facenet-pytorch'),
        ('ArcFace', 'Very High', 'Fast', '512D', 'insightface'),
        ('ResNet50', 'Medium', 'Slow', '2048D', 'tensorflow'),
        ('MobileNetV2', 'Medium', 'Very Fast', '1280D', 'tensorflow')
    ]
    
    for model_data in models_data:
        tree.insert('', 'end', values=model_data)
    
    tree.pack(fill='both', expand=True, padx=10, pady=10)

# Create main GUI
root = tk.Tk()
root.title("Enhanced Face Recognition Attendance System")
root.geometry("500x700")

# Model selection frame
model_frame = ttk.LabelFrame(root, text="Model Configuration", padding=10)
model_frame.pack(fill="x", padx=10, pady=5)

ttk.Label(model_frame, text="Select Model:").pack(anchor="w")
model_var = tk.StringVar(value="MobileNetV2")
model_combo = ttk.Combobox(model_frame, textvariable=model_var,
                          values=list(MODEL_CONFIGS.keys()), state="readonly")
model_combo.pack(fill="x", pady=2)

model_info_var = tk.StringVar(value="Click 'Initialize Model' to load")
ttk.Label(model_frame, textvariable=model_info_var, foreground="blue").pack(anchor="w", pady=2)

ttk.Button(model_frame, text="Initialize Model", command=initialize_model).pack(pady=5)
ttk.Button(model_frame, text="Compare Models", command=show_model_comparison).pack(pady=2)

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
threshold_var = tk.DoubleVar(value=0.7)
threshold_scale = ttk.Scale(settings_frame, from_=0.5, to=0.95, 
                           variable=threshold_var, orient="horizontal")
threshold_scale.pack(fill="x", pady=2)
threshold_label = ttk.Label(settings_frame, text="0.70")
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
ttk.Button(enroll_frame, text="Enroll Student (Enhanced)", 
          command=start_enhanced_enrollment).pack(pady=5)

# Attendance frame
attendance_frame = ttk.LabelFrame(root, text="Attendance Marking", padding=10)
attendance_frame.pack(fill="x", padx=10, pady=5)

ttk.Button(attendance_frame, text="Mark Attendance (Enhanced)", 
          command=start_enhanced_attendance).pack(pady=5)

# Info frame
info_frame = ttk.LabelFrame(root, text="System Information", padding=10)
info_frame.pack(fill="x", padx=10, pady=5)

info_text = f"""
Available Models:
• FaceNet: High accuracy, fast inference
• ArcFace: Highest accuracy, state-of-the-art
• ResNet50: Good accuracy, slower inference
• MobileNetV2: Balanced speed/accuracy

Features:
• MTCNN face detection (if available)
• Quality-based enrollment
• Performance monitoring
• Configurable similarity metrics
• Real-time FPS display
"""

ttk.Label(info_frame, text=info_text, justify="left").pack(anchor="w")

# Initialize with default model
initialize_model()

root.mainloop()