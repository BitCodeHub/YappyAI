#!/usr/bin/env python3
"""Check if PyPDF2 is installed"""

try:
    import PyPDF2
    print("✅ PyPDF2 is installed")
    print(f"Version: {PyPDF2.__version__ if hasattr(PyPDF2, '__version__') else 'Unknown'}")
except ImportError:
    print("❌ PyPDF2 is NOT installed")
    print("Installing PyPDF2...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    print("✅ PyPDF2 installed successfully!")