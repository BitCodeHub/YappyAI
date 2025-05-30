#!/usr/bin/env python3
"""
Quick test script to verify authentication is working
"""

import subprocess
import time
import requests
import json

def test_auth():
    print("Testing AgenticSeek Authentication System...")
    
    # Base URL
    base_url = "http://localhost:8000"
    
    # Test user credentials
    test_user = {
        "username": "testuser",
        "password": "testpass123"
    }
    
    print("\n1. Testing health endpoint (should work without auth)...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("Make sure the API server is running: python3 api.py")
        return
    
    print("\n2. Testing protected endpoint without auth (should fail)...")
    try:
        response = requests.get(f"{base_url}/is_active")
        print(f"Status: {response.status_code}")
        if response.status_code == 401:
            print("✅ Correctly rejected unauthorized request")
        else:
            print("❌ Expected 401, got:", response.status_code)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n3. Registering new user...")
    try:
        response = requests.post(f"{base_url}/auth/register", json=test_user)
        if response.status_code == 200:
            print(f"✅ User registered: {response.json()}")
        elif response.status_code == 400:
            print("ℹ️  User already exists")
        else:
            print(f"❌ Registration failed: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n4. Logging in...")
    try:
        response = requests.post(f"{base_url}/auth/login", json=test_user)
        if response.status_code == 200:
            token_data = response.json()
            token = token_data["access_token"]
            print(f"✅ Login successful!")
            print(f"   Token: {token[:20]}...")
            print(f"   Expires in: {token_data['expires_in']} seconds")
        else:
            print(f"❌ Login failed: {response.status_code} - {response.json()}")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print("\n5. Testing protected endpoint with auth...")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{base_url}/is_active", headers=headers)
        if response.status_code == 200:
            print(f"✅ Authorized request successful: {response.json()}")
        else:
            print(f"❌ Request failed: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n6. Testing query endpoint with auth...")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        query_data = {"query": "What is 2+2?"}
        response = requests.post(f"{base_url}/query", json=query_data, headers=headers)
        if response.status_code == 200:
            print(f"✅ Query successful!")
            result = response.json()
            print(f"   Agent: {result.get('agent_name', 'Unknown')}")
            print(f"   Answer: {result.get('answer', 'No answer')[:50]}...")
        else:
            print(f"❌ Query failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n✅ Authentication system test complete!")
    print("\nYou can now:")
    print("1. Run the frontend: cd frontend/agentic-seek-front && npm start")
    print("2. Use the example script: python3 examples/auth_example.py")
    print("3. Check users.json to see stored user data")

if __name__ == "__main__":
    test_auth()