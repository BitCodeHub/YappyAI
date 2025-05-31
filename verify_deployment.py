#!/usr/bin/env python3
"""Verify which version is deployed"""

import requests
import json

# Replace with your actual Render URL
RENDER_URL = "https://yappyai.onrender.com"  # Change this to your URL

def check_health():
    """Check the health endpoint to see version"""
    try:
        response = requests.get(f"{RENDER_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("🟢 Health Check Response:")
            print(json.dumps(data, indent=2))
            
            # Check version
            version = data.get("version", "Unknown")
            features = data.get("features", [])
            
            print(f"\n📌 Version: {version}")
            print(f"📋 Features: {features}")
            
            # Check for v4 specific features
            if "context_awareness" in features:
                print("✅ V4 is deployed (has context_awareness)")
            elif "real_time_search" in features:
                print("⚠️  V3 is deployed (has real_time_search but no context)")
            else:
                print("❌ Old version is deployed (no advanced features)")
                
            # Check date
            if "current_date" in data:
                print(f"📅 Server date: {data['current_date']}")
            
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error checking health: {e}")

def test_chat_endpoint():
    """Test if chat endpoint has context features"""
    print("\n🧪 Testing Chat Endpoint...")
    
    # You need to login first to get a token
    # This is just to show the structure
    print("To fully test, you need to:")
    print("1. Login via the web interface")
    print("2. Check browser DevTools Network tab")
    print("3. Look at the /api/chat response")
    print("4. Check for 'context_used' and 'web_searched' fields")

if __name__ == "__main__":
    print("🔍 Checking which version is deployed...")
    print(f"🌐 URL: {RENDER_URL}")
    print("-" * 50)
    
    check_health()
    test_chat_endpoint()
    
    print("\n💡 Quick Fixes:")
    print("1. Make sure you updated the Start Command in Render Settings")
    print("2. Check Render logs for any startup errors")
    print("3. Verify the deployment completed successfully")
    print("4. Try a manual deploy from Render dashboard")