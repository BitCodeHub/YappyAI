#!/usr/bin/env python3
"""Test browser screenshot functionality"""

import os
from PIL import Image, ImageDraw, ImageFont

# Create screenshots directory if it doesn't exist
os.makedirs('.screenshots', exist_ok=True)

# Create a test image to simulate a screenshot
img = Image.new('RGB', (1280, 720), color='white')
draw = ImageDraw.Draw(img)

# Add some text to make it clear this is a test
try:
    # Try to use a system font
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
except:
    # Fall back to default font
    font = ImageFont.load_default()

# Draw test message
draw.text((100, 300), "Browser View Test - No Active Browser Session", fill='black', font=font)
draw.text((100, 400), "Start a web search to see live screenshots", fill='gray', font=font)

# Save the test image
img.save('.screenshots/updated_screen.png')
print("Test screenshot created at .screenshots/updated_screen.png")

# Also check if we can access it via the API
import requests
try:
    response = requests.get('http://127.0.0.1:8000/screenshots/updated_screen.png')
    if response.status_code == 200:
        print("✓ Screenshot endpoint is accessible")
    else:
        print(f"✗ Screenshot endpoint returned status {response.status_code}")
except Exception as e:
    print(f"✗ Could not reach screenshot endpoint: {e}")