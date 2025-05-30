#!/usr/bin/env python3
"""
Complete setup and run script for AgenticSeek with PDF support
This script will:
1. Install all dependencies
2. Verify installation
3. Start the API server
"""

import subprocess
import sys
import os
import time

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n🔧 {description}...")
    try:
        subprocess.check_call(cmd, shell=True)
        print(f"✅ {description} - Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed: {e}")
        return False

def install_dependencies():
    """Install all required dependencies"""
    print("📦 Installing dependencies for AgenticSeek with PDF support...")
    print("=" * 60)
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # First, try to install all requirements
    if run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                   "Installing all requirements"):
        return True
    
    # If that fails, try installing key packages individually
    print("\n🔧 Attempting to install key packages individually...")
    key_packages = [
        'pypdf>=5.4.0',
        'matplotlib>=3.5.0', 
        'seaborn>=0.12.0',
        'pandas>=1.5.0',
        'google-generativeai>=0.3.0',
        'fastapi>=0.115.12',
        'uvicorn>=0.34.0',
        'aiofiles>=24.1.0'
    ]
    
    failed = []
    for package in key_packages:
        if not run_command(f"{sys.executable} -m pip install {package}", 
                          f"Installing {package}"):
            failed.append(package)
    
    if failed:
        print(f"\n⚠️  Failed to install: {', '.join(failed)}")
        return False
    
    return True

def verify_installation():
    """Verify key packages are installed"""
    print("\n🔍 Verifying installation...")
    
    required_modules = {
        'pypdf': 'PDF processing',
        'matplotlib': 'Chart generation',
        'seaborn': 'Advanced visualizations',
        'pandas': 'Data analysis',
        'google.generativeai': 'AI processing',
        'fastapi': 'Web framework',
        'uvicorn': 'ASGI server'
    }
    
    all_good = True
    for module, description in required_modules.items():
        try:
            if '.' in module:
                parts = module.split('.')
                __import__(parts[0])
            else:
                __import__(module)
            print(f"✅ {module} ({description}) - Installed")
        except ImportError:
            print(f"❌ {module} ({description}) - Not found")
            all_good = False
    
    return all_good

def start_api():
    """Start the API server"""
    print("\n🚀 Starting Neo API with PDF support...")
    print("=" * 60)
    print("📄 PDF files will be automatically analyzed when uploaded")
    print("📊 CSV/Excel files will generate visualizations")
    print("🌐 API will be available at http://localhost:8000")
    print("=" * 60)
    print("\n⏳ Starting server... (Press Ctrl+C to stop)")
    
    try:
        # Start the API
        subprocess.run([sys.executable, "api_clean.py"])
    except KeyboardInterrupt:
        print("\n\n✋ API stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting API: {e}")
        return False
    
    return True

def main():
    """Main setup and run function"""
    print("🎯 AgenticSeek Setup & Run Script")
    print("=" * 60)
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed. Please check the errors above.")
        print("You may need to:")
        print("1. Update pip: python -m pip install --upgrade pip")
        print("2. Use a virtual environment")
        print("3. Check Python version (3.8+ required)")
        return
    
    # Step 2: Verify installation
    if not verify_installation():
        print("\n⚠️  Some packages are missing. The API may not work correctly.")
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Step 3: Start API
    print("\n✅ All dependencies installed successfully!")
    time.sleep(2)  # Brief pause before starting
    
    start_api()

if __name__ == "__main__":
    main()