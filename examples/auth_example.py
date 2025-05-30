#!/usr/bin/env python3
"""
Example script demonstrating how to use the authentication system with AgenticSeek API.
"""

import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

def register_user(username: str, password: str):
    """Register a new user"""
    response = requests.post(
        f"{BASE_URL}/auth/register",
        json={"username": username, "password": password}
    )
    
    if response.status_code == 200:
        print(f"✅ User '{username}' registered successfully!")
        return True
    else:
        print(f"❌ Registration failed: {response.json()}")
        return False

def login_user(username: str, password: str):
    """Login and get access token"""
    response = requests.post(
        f"{BASE_URL}/auth/login",
        json={"username": username, "password": password}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Login successful!")
        print(f"   Access token: {data['access_token'][:20]}...")
        print(f"   Token expires in: {data['expires_in']} seconds")
        return data['access_token']
    else:
        print(f"❌ Login failed: {response.json()}")
        return None

def make_authenticated_request(token: str, query: str):
    """Make an authenticated request to the query endpoint"""
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    response = requests.post(
        f"{BASE_URL}/query",
        json={"query": query},
        headers=headers
    )
    
    if response.status_code == 200:
        print("✅ Query successful!")
        data = response.json()
        print(f"   Agent: {data.get('agent_name', 'Unknown')}")
        print(f"   Answer: {data.get('answer', 'No answer')[:100]}...")
        return data
    else:
        print(f"❌ Query failed: {response.json()}")
        return None

def check_current_user(token: str):
    """Check the current authenticated user"""
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    response = requests.get(
        f"{BASE_URL}/auth/me",
        headers=headers
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Current user: {data['username']}")
        return data['username']
    else:
        print(f"❌ Failed to get current user: {response.json()}")
        return None

def main():
    print("=== AgenticSeek Authentication Example ===\n")
    
    # Example credentials
    username = "test_user"
    password = "secure_password123"
    
    # Step 1: Register a new user
    print("1. Registering new user...")
    register_user(username, password)
    print()
    
    # Step 2: Login and get token
    print("2. Logging in...")
    token = login_user(username, password)
    print()
    
    if token:
        # Step 3: Check current user
        print("3. Checking current user...")
        check_current_user(token)
        print()
        
        # Step 4: Make an authenticated query
        print("4. Making authenticated query...")
        make_authenticated_request(token, "What is 2+2?")
        print()
        
        # Step 5: Try with invalid token
        print("5. Testing with invalid token...")
        headers = {"Authorization": "Bearer invalid_token"}
        response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")

if __name__ == "__main__":
    main()