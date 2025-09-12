"""
Benchmark script to compare model performance
"""
import time
import cv2
import numpy as np
import sys
import os

# Add parent directory to path to import from enhanced_app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_app import FaceRecognitionModel, PerformanceMonitor
except ImportError:
    print("Please ensure enhanced_app.py is in the same directory")
    sys.exit(1)

def benchmark_model(model_name, num_frames=100, image_size=(640, 480)):
    """Benchmark a specific model"""
    print(f"\n🔍 Benchmarking {model_name}")
    print("-" * 40)
    
    try:
        # Initialize model
        model = FaceRecognitionModel(model_name)
        monitor = PerformanceMonitor()
        
        # Generate test frames (simulating camera input)
        test_frames = []
        for i in range(num_frames):
            # Create random frame
            frame = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            # Add a simple face-like rectangle
            h, w = image_size
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (128, 128, 128), -1)
            test_frames.append(frame)
        
        # Benchmark face detection
        detection_times = []
        total_faces = 0
        
        for frame in test_frames:
            start_time = time.time()
            faces = model.detect_faces(frame)
            detection_time = time.time() - start_time
            detection_times.append(detection_time)
            total_faces += len(faces)
        
        # Benchmark face recognition (using first detected face)
        recognition_times = []
        embeddings = []
        
        for frame in test_frames[:10]:  # Test on first 10 frames
            faces = model.detect_faces(frame)
            if faces:
                x, y, w, h = faces[0]
                face_img = frame[y:y+h, x:x+w]
                if face_img.size > 0:
                    start_time = time.time()
                    embedding, proc_time = model.get_embedding(face_img)
                    recognition_time = time.time() - start_time
                    recognition_times.append(recognition_time)
                    embeddings.append(embedding)
        
        # Calculate statistics
        avg_detection_time = np.mean(detection_times) * 1000  # Convert to ms
        avg_recognition_time = np.mean(recognition_times) * 1000 if recognition_times else 0
        
        detection_fps = 1.0 / (np.mean(detection_times)) if detection_times else 0
        recognition_fps = 1.0 / (np.mean(recognition_times)) if recognition_times else 0
        
        embedding_size = len(embeddings[0]) if embeddings else 0
        
        # Print results
        print(f"✅ Model: {model.model_name}")
        print(f"📊 Detection Performance:")
        print(f"   • Average time: {avg_detection_time:.2f} ms")
        print(f"   • FPS: {detection_fps:.1f}")
        print(f"   • Faces detected: {total_faces}/{num_frames}")
        
        if recognition_times:
            print(f"🧠 Recognition Performance:")
            print(f"   • Average time: {avg_recognition_time:.2f} ms")
            print(f"   • FPS: {recognition_fps:.1f}")
            print(f"   • Embedding size: {embedding_size}D")
        
        return {
            'model': model.model_name,
            'detection_time_ms': avg_detection_time,
            'detection_fps': detection_fps,
            'recognition_time_ms': avg_recognition_time,
            'recognition_fps': recognition_fps,
            'embedding_size': embedding_size,
            'faces_detected': total_faces
        }
        
    except Exception as e:
        print(f"❌ Error benchmarking {model_name}: {str(e)}")
        return None

def main():
    print("🚀 Enhanced Face Recognition - Performance Benchmark")
    print("=" * 60)
    
    # Models to test
    models_to_test = [
        "MobileNetV2",
        "ResNet50", 
        "FaceNet",
        "ArcFace"
    ]
    
    results = []
    
    for model_name in models_to_test:
        result = benchmark_model(model_name, num_frames=50)
        if result:
            results.append(result)
    
    # Summary comparison
    if results:
        print(f"\n📋 PERFORMANCE COMPARISON")
        print("=" * 60)
        print(f"{'Model':<12} {'Det.FPS':<8} {'Rec.FPS':<8} {'Det.Time':<10} {'Rec.Time':<10} {'Emb.Size':<8}")
        print("-" * 60)
        
        for result in results:
            print(f"{result['model']:<12} "
                  f"{result['detection_fps']:<8.1f} "
                  f"{result['recognition_fps']:<8.1f} "
                  f"{result['detection_time_ms']:<10.1f} "
                  f"{result['recognition_time_ms']:<10.1f} "
                  f"{result['embedding_size']:<8}D")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 30)
        
        fastest_detection = min(results, key=lambda x: x['detection_time_ms'])
        fastest_recognition = min(results, key=lambda x: x['recognition_time_ms'] or float('inf'))
        smallest_embedding = min(results, key=lambda x: x['embedding_size'])
        
        print(f"🏃 Fastest Detection: {fastest_detection['model']} ({fastest_detection['detection_fps']:.1f} FPS)")
        print(f"🧠 Fastest Recognition: {fastest_recognition['model']} ({fastest_recognition['recognition_fps']:.1f} FPS)")
        print(f"💾 Smallest Embeddings: {smallest_embedding['model']} ({smallest_embedding['embedding_size']}D)")
        
        # Usage recommendations
        print(f"\n🎯 USAGE RECOMMENDATIONS")
        print("-" * 25)
        print(f"🖥️  Real-time applications: MobileNetV2")
        print(f"📱 Mobile/Edge devices: MobileNetV2")
        print(f"🎯 High accuracy needed: ArcFace or FaceNet")
        print(f"⚡ Balanced performance: FaceNet")
        print(f"🔄 Legacy compatibility: ResNet50")

if __name__ == "__main__":
    main()