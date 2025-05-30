#!/usr/bin/env python3
"""
Minimal API server for testing authentication
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our auth module
from sources.auth import AuthManager, UserCredentials, TokenResponse

# Create app and auth manager
app = FastAPI(title="AgenticSeek Auth Test")
auth_manager = AuthManager()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    agent_name: str = "TestAgent"

# Public endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

# Auth endpoints
@app.post("/auth/register")
async def register(credentials: UserCredentials):
    return auth_manager.register_user(credentials.username, credentials.password)

@app.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserCredentials):
    username = auth_manager.authenticate_user(credentials.username, credentials.password)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = auth_manager.create_access_token(username)
    return TokenResponse(
        access_token=token,
        expires_in=auth_manager.token_expiry_hours * 3600
    )

@app.get("/auth/me")
async def get_current_user(current_user: str = Depends(auth_manager.get_current_user)):
    return {"username": current_user}

# Protected endpoints
@app.get("/is_active")
async def is_active(current_user: str = Depends(auth_manager.get_current_user)):
    return {"is_active": True, "user": current_user}

@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest, 
    current_user: str = Depends(auth_manager.get_current_user)
):
    # Simple mock response
    if "2+2" in request.query:
        answer = "4"
    else:
        answer = f"I received your query: '{request.query}'"
    
    return QueryResponse(
        answer=answer,
        agent_name=f"TestAgent (user: {current_user})"
    )

if __name__ == "__main__":
    print("Starting minimal API server with authentication...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)