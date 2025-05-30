#!/usr/bin/env python3
"""
Install all dependencies for AgenticSeek with PDF support
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a single package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("📦 Installing dependencies for AgenticSeek with PDF support...")
    print("=" * 50)
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found!")
        sys.exit(1)
    
    # Read requirements
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"📋 Found {len(requirements)} packages to install")
    
    # Key packages for PDF support
    pdf_packages = ['pypdf', 'matplotlib', 'seaborn', 'pandas', 'google-generativeai']
    
    # Install all requirements
    print("\n🔧 Installing all requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Some packages failed to install: {e}")
        print("\n🔧 Attempting to install key PDF packages individually...")
        
        # Try installing key packages one by one
        for package in pdf_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed")
            else:
                print(f"❌ Failed to install {package}")
    
    # Verify key packages
    print("\n🔍 Verifying key PDF support packages...")
    installed = []
    missing = []
    
    for package in pdf_packages:
        try:
            __import__(package.replace('-', '_').split('>=')[0])
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    if installed:
        print("\n✅ Installed packages:")
        for pkg in installed:
            print(f"  - {pkg}")
    
    if missing:
        print("\n❌ Missing packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nTry installing missing packages manually:")
        print(f"pip install {' '.join(missing)}")
    else:
        print("\n🎉 All PDF support packages are ready!")
        print("\n🚀 You can now start the API with:")
        print("  python api_clean.py")
        print("  OR")
        print("  python start_neo_api.py")

if __name__ == "__main__":
    main()