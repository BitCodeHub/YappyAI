#!/usr/bin/env python3
"""Create a simple placeholder screenshot"""

import os

# Create screenshots directory if it doesn't exist
os.makedirs('.screenshots', exist_ok=True)

# Create a simple placeholder by copying any existing image or creating an empty file
placeholder_path = '.screenshots/updated_screen.png'

# For now, just create an empty file - the frontend will show placeholder.png
with open(placeholder_path, 'wb') as f:
    # Write a minimal PNG header to make it a valid (tiny) PNG file
    # This is a 1x1 transparent PNG
    png_data = bytes.fromhex('89504e470d0a1a0a0000000d4948445200000001000000010100000000376ef9240000000a49444154789c626001000000050001a5f645400000000049454e44ae426082')
    f.write(png_data)

print(f"Created placeholder at {placeholder_path}")

# Check the API endpoint
import requests
try:
    # First check if API is running
    health = requests.get('http://127.0.0.1:8000/health')
    print(f"API health check: {health.status_code}")
    
    # Try the screenshot endpoint
    response = requests.get('http://127.0.0.1:8000/screenshots/updated_screen.png')
    print(f"Screenshot endpoint status: {response.status_code}")
except Exception as e:
    print(f"API might not be running: {e}")