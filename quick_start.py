#!/usr/bin/env python3
"""Quick start script - assumes dependencies are installed"""

import subprocess
import sys
import os

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("🚀 Starting Neo API with PDF support...")
print("=" * 50)
print("📄 PDF upload and analysis enabled")
print("📊 Data visualization enabled")
print("🌐 API at http://localhost:8000")
print("=" * 50)
print("\nPress Ctrl+C to stop\n")

try:
    subprocess.run([sys.executable, "api_clean.py"])
except KeyboardInterrupt:
    print("\n✋ Stopped")