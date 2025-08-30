#!/usr/bin/env python3
"""
Urban Air Quality Policy Assistant - Streamlit Runner

This script launches the Streamlit application for the Urban Air Quality Policy Assistant.
Run this file to start the web application.
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages if not already installed"""
    try:
        import streamlit
        import plotly
        import pandas
        import numpy
        print("✅ All required packages are already installed")
    except ImportError as e:
        print(f"📦 Installing missing package: {e.name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_streamlit():
    """Launch the Streamlit application"""
    print("🚀 Starting Urban Air Quality Policy Assistant...")
    print("🌍 Open your browser to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    print("🌍 Urban Air Quality Policy Assistant")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("❌ Error: streamlit_app.py not found in current directory")
        print("Please run this script from the urban-air-forecast directory")
        sys.exit(1)
    
    # Install requirements and run
    install_requirements()
    run_streamlit()