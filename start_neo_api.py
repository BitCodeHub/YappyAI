#!/usr/bin/env python3
"""
Start Neo API with PDF support
This replaces the original api.py with api_clean.py for enhanced file upload features
"""

import subprocess
import sys
import os

def main():
    # Change to the project directory
    os.chdir('/Users/jimmylam/Downloads/agenticSeek-main')
    
    print("🚀 Starting Neo API with PDF support...")
    print("📄 PDF files will be automatically analyzed when uploaded")
    print("📊 CSV/Excel files will generate visualizations")
    print("🌐 API will be available at http://localhost:8000")
    print("-" * 50)
    
    try:
        # Run the clean API with PDF support
        subprocess.run([sys.executable, 'api_clean.py'])
    except KeyboardInterrupt:
        print("\n✋ API stopped by user")
    except Exception as e:
        print(f"❌ Error starting API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()