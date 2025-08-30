#!/usr/bin/env python3
"""
Air Quality Image Analyzer - Launcher Script

This script launches the image-based air quality analysis application.
"""

import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'opencv-python',
        'plotly',
        'Pillow',
        'numpy',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'Pillow':
                from PIL import Image
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("✅ All packages installed successfully!")
    else:
        print("✅ All required packages are available")

def run_image_analyzer():
    """Launch the image analysis Streamlit application"""
    print("📸 Starting Air Quality Image Analyzer...")
    print("🌍 Open your browser to: http://localhost:8507")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 60)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "modern_air_analyzer.py",
            "--server.port", "8507",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    print("📸 Air Quality Image Analyzer")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("modern_air_analyzer.py").exists():
        print("❌ Error: modern_air_analyzer.py not found in current directory")
        print("Please run this script from the urban-air-forecast directory")
        sys.exit(1)
    
    # Check requirements and run
    check_requirements()
    run_image_analyzer()