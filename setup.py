#!/usr/bin/env python3
"""
Installation script for enhanced face recognition attendance system
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("Enhanced Face Recognition Attendance System - Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher required")
        return
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Read requirements
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print(f"❌ {req_file} not found")
        return
    
    with open(req_file, 'r') as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"\n📦 Installing {len(packages)} packages...")
    
    # Core packages (required)
    core_packages = ['opencv-python', 'tensorflow', 'numpy', 'scikit-learn', 'Pillow']
    optional_packages = ['mtcnn', 'facenet-pytorch', 'insightface']
    
    # Install core packages
    failed_core = []
    for package in core_packages:
        matching_req = next((p for p in packages if package in p), package)
        print(f"Installing {matching_req}...")
        if not install_package(matching_req):
            failed_core.append(package)
            print(f"❌ Failed to install {package}")
        else:
            print(f"✅ Installed {package}")
    
    if failed_core:
        print(f"\n❌ Failed to install core packages: {failed_core}")
        print("Please install them manually before proceeding")
        return
    
    # Install optional packages
    failed_optional = []
    for package in optional_packages:
        matching_req = next((p for p in packages if package in p), package)
        print(f"Installing {matching_req} (optional)...")
        if not install_package(matching_req):
            failed_optional.append(package)
            print(f"⚠️  Failed to install {package} (optional)")
        else:
            print(f"✅ Installed {package}")
    
    print("\n" + "=" * 50)
    print("📋 INSTALLATION SUMMARY")
    print("=" * 50)
    
    if not failed_core:
        print("✅ All core packages installed successfully!")
        
        print("\n🔧 AVAILABLE MODELS:")
        print("• ResNet50: ✅ (tensorflow)")
        print("• MobileNetV2: ✅ (tensorflow)")
        
        if 'mtcnn' not in failed_optional:
            print("• MTCNN Face Detection: ✅")
        else:
            print("• MTCNN Face Detection: ❌ (will use Haar cascades)")
        
        if 'facenet-pytorch' not in failed_optional:
            print("• FaceNet: ✅")
        else:
            print("• FaceNet: ❌")
        
        if 'insightface' not in failed_optional:
            print("• ArcFace/InsightFace: ✅")
        else:
            print("• ArcFace/InsightFace: ❌")
        
        print(f"\n🚀 You can now run:")
        print(f"   python enhanced_app.py")
        
        if failed_optional:
            print(f"\n⚠️  Some optional packages failed to install:")
            print(f"   {', '.join(failed_optional)}")
            print(f"   The system will work with fallback models.")
    else:
        print("❌ Core package installation failed. Please check your Python environment.")

if __name__ == "__main__":
    main()