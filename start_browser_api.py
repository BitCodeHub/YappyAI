#!/usr/bin/env python3
"""
Simplified API starter that enables browser functionality
This bypasses some dependencies that have Python version issues
"""

import subprocess
import sys
import os

# Set environment variable
os.environ["GOOGLE_API_KEY"] = "AIzaSyBj_AXSeaKBSazrWnx5NAsKm32k8sYjkxk"

# Install only essential dependencies
essential_deps = [
    "fastapi",
    "uvicorn",
    "aiofiles",
    "python-multipart",
    "selenium",
    "beautifulsoup4",
    "markdownify",
    "fake-useragent",
    "chromedriver-autoinstaller",
    "undetected-chromedriver",
    "selenium-stealth"
]

print("Installing essential dependencies...")
for dep in essential_deps:
    subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("\nTo enable browser view in AgenticSeek:")
print("1. The full api.py needs to run (not api_gemini.py)")
print("2. You need Chrome/Chromium installed")
print("3. Use queries that require web browsing like:")
print('   - "search the web for Python tutorials"')
print('   - "browse to google.com and search for flights"')
print('   - "find flights on the web from LAX to Buffalo"')
print("\nThe browser view only activates when the Browser Agent is used.")
print("\nAlternatively, you can use api_working.py which has full functionality.")