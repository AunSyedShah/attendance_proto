import cv2
import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import subprocess
import platform
from datetime import datetime
from sklearn.preprocessing import normalize
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------- CONFIG ----------------
DATA_DIR = "data"
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.pkl")
ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- MODEL LOADING ----------------
def load_model(model_name="ResNet50"):
    """
    Load embedding model. Can be extended for other backbones.
    """
    if model_name == "ResNet50":
        return ResNet50(weights="imagenet", include_top=False, pooling="avg")
    # TODO: Add other models (e.g., MobileNet, EfficientNet, custom ArcFace)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

model = load_model("ResNet50")

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

# ---------------- ENROLLMENT ----------------
def enroll_student(student_id, cam_index, max_samples=20):
    embeddings = load_embeddings()
    cap = cv2.VideoCapture(cam_index)

    collected = []
    print(f"Enrolling {student_id}... Press 'q' to stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            embedding = get_embedding(face)
            collected.append(embedding)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Enrollment", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or len(collected) >= max_samples:
            break

    cap.release()
    cv2.destroyAllWindows()

    if collected:
        embeddings[student_id] = np.mean(collected, axis=0)  # mean embedding
        save_embeddings(embeddings)
        messagebox.showinfo("Success", f"Enrolled {student_id} with {len(collected)} samples")
    else:
        messagebox.showwarning("Failed", f"No faces captured for {student_id}")

# ---------------- ATTENDANCE ----------------
def mark_attendance(cam_index, threshold=0.6):
    embeddings = load_embeddings()
    if not embeddings:
        messagebox.showerror("Error", "No enrolled students found!")
        return

    cap = cv2.VideoCapture(cam_index)
    attendance = set()

    print("Marking attendance... Press 'q' to stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            emb = get_embedding(face)

            # Compare embeddings
            best_match, best_dist = "Unknown", float("inf")
            for sid, stored_emb in embeddings.items():
                dist = np.linalg.norm(stored_emb - emb)
                if dist < best_dist:
                    best_match, best_dist = sid, dist

            label = best_match if best_dist < threshold else "Unknown"
            if label != "Unknown":
                attendance.add(label)

            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        cv2.imshow("Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if attendance:
        with open(ATTENDANCE_FILE, "a") as f:
            for sid in attendance:
                f.write(f"{sid},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        messagebox.showinfo("Done", f"Attendance: {', '.join(attendance)}")
    else:
        messagebox.showwarning("No Faces", "No attendance recorded")

# ---------------- GUI ----------------
def start_enrollment():
    sid = student_id_var.get().strip()
    if not sid:
        messagebox.showerror("Error", "Enter Student ID")
        return
    cam_index = int(camera_var.get().split(":")[0])
    enroll_student(sid, cam_index)

def start_attendance():
    cam_index = int(camera_var.get().split(":")[0])
    mark_attendance(cam_index)

root = tk.Tk()
root.title("Face Recognition Attendance System")

# Camera dropdown
cameras = get_available_cameras()
camera_var = tk.StringVar(value=f"{cameras[0][0]}: {cameras[0][1]}")
ttk.Label(root, text="Select Camera:").pack(pady=5)
camera_dropdown = ttk.Combobox(root, textvariable=camera_var,
                               values=[f"{idx}: {name}" for idx, name in cameras],
                               state="readonly")
camera_dropdown.pack(pady=5)

# Enrollment
student_id_var = tk.StringVar()
ttk.Label(root, text="Student ID:").pack(pady=5)
ttk.Entry(root, textvariable=student_id_var).pack(pady=5)
ttk.Button(root, text="Enroll Student", command=start_enrollment).pack(pady=10)

# Attendance
ttk.Button(root, text="Mark Attendance", command=start_attendance).pack(pady=10)

root.mainloop()
