#!/usr/bin/env python3

import jwt
import bcrypt
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

class UserCredentials(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    username: str
    created_at: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class AuthManager:
    def __init__(self, secret_key: str = None, token_expiry_hours: int = 24):
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
        self.token_expiry_hours = token_expiry_hours
        self.algorithm = "HS256"
        self.users_file = "users.json"
        self.security = HTTPBearer()
        
        # Create users file if it doesn't exist
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump({}, f)
    
    def _load_users(self) -> Dict:
        """Load users from JSON file"""
        with open(self.users_file, 'r') as f:
            return json.load(f)
    
    def _save_users(self, users: Dict):
        """Save users to JSON file"""
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def create_access_token(self, username: str) -> str:
        """Create a JWT access token"""
        expire = datetime.now(timezone.utc) + timedelta(hours=self.token_expiry_hours)
        to_encode = {
            "sub": username,
            "exp": expire,
            "iat": datetime.now(timezone.utc)
        }
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def decode_token(self, token: str) -> Optional[str]:
        """Decode and validate a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            if username is None:
                return None
            return username
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def register_user(self, username: str, password: str) -> UserResponse:
        """Register a new user"""
        users = self._load_users()
        
        if username in users:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        hashed_password = self.hash_password(password)
        users[username] = {
            "password": hashed_password,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        self._save_users(users)
        
        return UserResponse(
            username=username,
            created_at=users[username]["created_at"]
        )
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate a user and return username if valid"""
        users = self._load_users()
        
        if username not in users:
            return None
        
        user = users[username]
        if not self.verify_password(password, user["password"]):
            return None
        
        return username
    
    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())) -> str:
        """Dependency to get the current authenticated user"""
        token = credentials.credentials
        username = self.decode_token(token)
        
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify user still exists
        users = self._load_users()
        if username not in users:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return username