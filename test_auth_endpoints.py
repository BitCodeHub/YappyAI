#!/usr/bin/env python3
"""
Test authentication endpoints
"""

import requests
import json

def test_auth_endpoints():
    base_url = "http://127.0.0.1:8000"
    
    print("🔍 Testing Yappy API authentication endpoints...")
    print("=" * 50)
    
    # Test health endpoint first
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("💡 Make sure the API is running: python3 api_clean.py")
        return
    
    # Test registration endpoint
    try:
        test_user = {
            "username": "testuser",
            "password": "testpass123"
        }
        
        response = requests.post(f"{base_url}/auth/register", json=test_user)
        print(f"📝 Registration test: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✅ Registration successful: {response.json()}")
        elif response.status_code == 400:
            print(f"⚠️  User might already exist: {response.json()}")
        else:
            print(f"❌ Registration failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Registration test failed: {e}")
    
    # Test login endpoint
    try:
        response = requests.post(f"{base_url}/auth/login", json=test_user)
        print(f"🔐 Login test: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Login successful: Got token")
            token = data.get('access_token')
            
            # Test authenticated endpoint
            headers = {"Authorization": f"Bearer {token}"}
            me_response = requests.get(f"{base_url}/auth/me", headers=headers)
            print(f"👤 User info test: {me_response.status_code} - {me_response.json()}")
            
        else:
            print(f"❌ Login failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Login test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Authentication test completed!")

if __name__ == "__main__":
    test_auth_endpoints()