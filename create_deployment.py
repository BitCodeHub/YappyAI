#!/usr/bin/env python3
"""
Create a deployment package for Yappy with PDF support
"""

import os
import shutil
import zipfile
import platform

def create_start_scripts(deploy_dir):
    """Create platform-specific start scripts"""
    
    # Windows batch file
    with open(os.path.join(deploy_dir, 'start_windows.bat'), 'w') as f:
        f.write("""@echo off
echo Starting Yappy with PDF Support...
echo =====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

REM Check if dependencies are installed
pip show pypdf >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Start the API server
echo Starting API server...
start cmd /k python api_clean.py

REM Wait for server to start
timeout /t 5 /nobreak >nul

REM Open browser
echo Opening browser...
start http://localhost:3000

echo.
echo Yappy is now running!
echo - API: http://localhost:8000
echo - Frontend: http://localhost:3000
echo.
echo Press any key to close this window (server will keep running)...
pause >nul
""")
    
    # Mac/Linux shell script
    with open(os.path.join(deploy_dir, 'start_unix.sh'), 'w') as f:
        f.write("""#!/bin/bash

echo "Starting Yappy with PDF Support..."
echo "====================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ first"
    exit 1
fi

# Check if dependencies are installed
if ! python3 -c "import pypdf" &> /dev/null; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Start the API server in background
echo "Starting API server..."
python3 api_clean.py &
API_PID=$!

# Wait for server to start
sleep 5

# Open browser
echo "Opening browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:3000
else
    xdg-open http://localhost:3000 2>/dev/null || echo "Please open http://localhost:3000 in your browser"
fi

echo
echo "Yappy is now running!"
echo "- API: http://localhost:8000"
echo "- Frontend: http://localhost:3000"
echo
echo "Press Enter to stop the server..."
read

# Stop the API server
kill $API_PID 2>/dev/null
echo "Server stopped."
""")
    
    # Make shell script executable
    os.chmod(os.path.join(deploy_dir, 'start_unix.sh'), 0o755)

def create_deployment():
    """Create a deployment package"""
    
    print("📦 Creating deployment package for Yappy...")
    
    # Create deployment directory
    deploy_dir = "yappy_deploy"
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(deploy_dir)
    
    # Files to copy
    files_to_copy = [
        'api_clean.py',
        'requirements.txt',
        'start_neo_api.py',
        'PDF_UPLOAD_GUIDE.md',
        'check_dependencies.py'
    ]
    
    # Copy files
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy(file, deploy_dir)
            print(f"✓ Copied {file}")
    
    # Copy directories (if they exist)
    dirs_to_copy = ['frontend', 'sources', 'prompts', 'searxng']
    for dir_name in dirs_to_copy:
        if os.path.exists(dir_name):
            shutil.copytree(dir_name, os.path.join(deploy_dir, dir_name))
            print(f"✓ Copied {dir_name}/")
    
    # Create .env template
    with open(os.path.join(deploy_dir, '.env'), 'w') as f:
        f.write("""# Yappy Configuration
# Add your Google Gemini API key here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Change ports if needed
API_PORT=8000
FRONTEND_PORT=3000
""")
    
    # Create comprehensive README
    with open(os.path.join(deploy_dir, 'README.txt'), 'w') as f:
        f.write("""Yappy with PDF Support - Installation Guide
==============================================

REQUIREMENTS:
- Python 3.8 or higher
- Google Gemini API key (get one at https://makersuite.google.com/app/apikey)

QUICK START:

Windows Users:
1. Double-click 'start_windows.bat'
2. Follow the prompts

Mac/Linux Users:
1. Open Terminal in this folder
2. Run: ./start_unix.sh

MANUAL INSTALLATION:

1. Install Python 3.8+ from https://python.org

2. Open terminal/command prompt in this directory

3. Install dependencies:
   pip install -r requirements.txt

4. Edit the .env file and add your Gemini API key

5. Start the application:
   python start_neo_api.py
   OR
   python api_clean.py

6. Open http://localhost:3000 in your browser

FEATURES:
- Upload and analyze PDF files
- Ask questions about PDF content
- Generate visualizations from CSV/Excel files
- 100% local and private

TROUBLESHOOTING:
- If port 8000 is in use: python api_clean.py 8001
- Check dependencies: python check_dependencies.py
- See PDF_UPLOAD_GUIDE.md for detailed instructions

For more help, see the documentation files included.
""")
    
    # Create start scripts
    create_start_scripts(deploy_dir)
    
    # Create simple installer script
    with open(os.path.join(deploy_dir, 'install.py'), 'w') as f:
        f.write("""#!/usr/bin/env python3
import subprocess
import sys
import os

print("Installing Yappy dependencies...")
print("=" * 40)

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("\\n✅ Installation complete!")
    print("\\nNow run one of these to start:")
    if os.name == 'nt':
        print("  start_windows.bat")
    else:
        print("  ./start_unix.sh")
except Exception as e:
    print(f"\\n❌ Installation failed: {e}")
    print("Try running: pip install -r requirements.txt")
""")
    
    # Create zip file
    zip_name = 'yappy_pdf_deploy.zip'
    print(f"\n📦 Creating {zip_name}...")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(deploy_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, deploy_dir)
                zipf.write(file_path, arcname)
    
    # Get file size
    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    
    print(f"\n✅ Deployment package created successfully!")
    print(f"📦 File: {zip_name} ({size_mb:.1f} MB)")
    print(f"📁 Also created folder: {deploy_dir}/")
    print("\nTo deploy to someone:")
    print("1. Send them the zip file")
    print("2. Tell them to extract it and run the start script")
    print("3. They'll need to add their own Gemini API key")

if __name__ == "__main__":
    create_deployment()