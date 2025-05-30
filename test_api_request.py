#!/usr/bin/env python3
"""
Test API request with CSV data
"""

import requests
import json
import base64

# Create test CSV content
csv_content = """Name,Age,Salary,Department
John,25,50000,Engineering
Mary,30,60000,Marketing
Bob,35,70000,Engineering
Alice,28,55000,Marketing
Charlie,32,65000,Engineering"""

# Make API request
url = "http://localhost:8000/query"
payload = {
    "query": "analyze this data and show me charts",
    "file_data": {
        "name": "test_data.csv",
        "type": "text/csv",
        "content": csv_content
    }
}

try:
    print("Sending test request...")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Request successful!")
        print(f"Response status: {data.get('status')}")
        answer = data.get('answer', '')
        print(f"Answer length: {len(answer)}")
        
        # Check for base64 images
        if 'data:image/png;base64,' in answer:
            print("✅ Found base64 image in response!")
            print(f"Number of images: {answer.count('data:image/png;base64,')}")
        else:
            print("❌ No base64 images found in response")
            print("First 500 chars of answer:")
            print(answer[:500])
    else:
        print(f"❌ Request failed: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Error: {e}")