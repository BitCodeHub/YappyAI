#!/usr/bin/env python3
"""
Check what API endpoints are available
"""

import requests
import json

def check_api():
    base_url = "http://127.0.0.1:8000"
    
    print("🔍 Checking Yappy API endpoints...")
    print("=" * 50)
    
    # List of endpoints to test
    endpoints = [
        ("GET", "/health", "Health check"),
        ("GET", "/docs", "API documentation"),
        ("POST", "/auth/register", "Registration"),
        ("POST", "/auth/login", "Login"),
        ("GET", "/auth/me", "User info"),
        ("POST", "/query", "Main query"),
        ("GET", "/latest_answer", "Latest answer"),
        ("GET", "/is_active", "Is active"),
        ("GET", "/stop", "Stop")
    ]
    
    for method, endpoint, description in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=2)
            elif method == "POST":
                # For POST endpoints, just check if they exist (will likely get 422 for missing data)
                response = requests.post(url, json={}, timeout=2)
            
            if response.status_code == 404:
                print(f"❌ {method} {endpoint} - {description}: NOT FOUND")
            elif response.status_code in [200, 422, 401]:  # 422 = validation error (endpoint exists)
                print(f"✅ {method} {endpoint} - {description}: Available")
            else:
                print(f"⚠️  {method} {endpoint} - {description}: Status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to {base_url}")
            print("💡 Make sure the API is running: python3 api_clean.py")
            break
        except Exception as e:
            print(f"❌ {method} {endpoint} - {description}: Error - {e}")
    
    print("\n" + "=" * 50)
    
    # Try to get API info
    try:
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("📖 API docs available at: http://127.0.0.1:8000/docs")
        else:
            print("📖 API docs not available")
    except:
        pass
    
    # Check if this is the clean API
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ API is responding")
            print("\n🔧 If registration still fails:")
            print("1. Stop the current API (Ctrl+C in the terminal)")
            print("2. Restart with: python3 api_clean.py")
            print("3. Try registration again")
        else:
            print("❌ API health check failed")
    except:
        print("❌ API is not responding")

if __name__ == "__main__":
    check_api()