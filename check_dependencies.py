#!/usr/bin/env python3
"""
Check if all required dependencies are installed for AgenticSeek PDF support
"""

import sys
import os

def check_module(module_name, display_name=None):
    """Check if a module is installed"""
    if display_name is None:
        display_name = module_name
    
    try:
        if module_name == 'google.generativeai':
            import google.generativeai
        else:
            __import__(module_name)
        print(f"✅ {display_name:<25} - Installed")
        return True
    except ImportError:
        print(f"❌ {display_name:<25} - Not installed")
        return False

def check_file(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description:<25} - Found")
        return True
    else:
        print(f"❌ {description:<25} - Not found")
        return False

def main():
    print("🔍 Checking AgenticSeek Dependencies")
    print("=" * 50)
    
    # Check Python version
    print(f"\n📌 Python Version: {sys.version}")
    
    # Check required modules
    print("\n📦 Required Packages:")
    modules = [
        ('pypdf', 'pypdf (PDF processing)'),
        ('matplotlib', 'matplotlib (Charts)'),
        ('seaborn', 'seaborn (Visualizations)'),
        ('pandas', 'pandas (Data analysis)'),
        ('google.generativeai', 'google-generativeai (AI)'),
        ('fastapi', 'fastapi (Web framework)'),
        ('uvicorn', 'uvicorn (ASGI server)'),
        ('aiofiles', 'aiofiles (Async files)'),
        ('numpy', 'numpy (Numerical)'),
        ('PIL', 'Pillow (Image processing)')
    ]
    
    installed = 0
    missing = 0
    
    for module, display in modules:
        if check_module(module, display):
            installed += 1
        else:
            missing += 1
    
    # Check important files
    print("\n📁 Required Files:")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = [
        (os.path.join(base_dir, 'api_clean.py'), 'api_clean.py'),
        (os.path.join(base_dir, 'requirements.txt'), 'requirements.txt'),
        (os.path.join(base_dir, 'start_neo_api.py'), 'start_neo_api.py')
    ]
    
    for filepath, description in files:
        check_file(filepath, description)
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Summary: {installed} installed, {missing} missing")
    
    if missing == 0:
        print("\n✅ All dependencies are installed!")
        print("\n🚀 You can start the API with:")
        print("   python api_clean.py")
        print("   OR")
        print("   python start_neo_api.py")
    else:
        print("\n⚠️  Some dependencies are missing!")
        print("\n📝 To install missing packages, run:")
        print("   pip install -r requirements.txt")
        print("\n📝 Or install individually:")
        for module, display in modules:
            try:
                __import__(module)
            except ImportError:
                print(f"   pip install {module}")

if __name__ == "__main__":
    main()