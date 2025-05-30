#!/usr/bin/env python3
"""
Working API server that integrates authentication with AgenticSeek frontend
"""

import os
import sys
import uuid
import asyncio
from typing import Dict
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# Import auth system
from sources.auth import AuthManager, UserCredentials, TokenResponse

# Create app
app = FastAPI(title="AgenticSeek API", version="0.1.0")
auth_manager = AuthManager()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
is_generating = False
query_resp_history = []
mock_agent_responses = {
    "capital": "The capital of France is Paris. It has been the capital since 987 AD and is known for landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.",
    "math": "2 + 2 = 4",
    "hello": "Hello! I'm Friday, your AI assistant. I can help you with various tasks like answering questions, browsing the web, writing code, and managing files. How can I assist you today?",
    "default": "I understand your question. Let me help you with that."
}

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    tts_enabled: bool = False

class QueryResponse(BaseModel):
    done: str = "false"
    answer: str = ""
    reasoning: str = ""
    agent_name: str = "Friday"
    success: str = "true"
    blocks: Dict = {}
    status: str = "Ready"
    uid: str = ""
    
    def jsonify(self):
        return {
            "done": self.done,
            "answer": self.answer,
            "reasoning": self.reasoning,
            "agent_name": self.agent_name,
            "success": self.success,
            "blocks": self.blocks,
            "status": self.status,
            "uid": self.uid
        }

# Authentication endpoints
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

# Public endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

# Protected endpoints
@app.get("/screenshot")
async def get_screenshot(current_user: str = Depends(auth_manager.get_current_user)):
    # Return a placeholder or actual screenshot if available
    screenshot_path = ".screenshots/updated_screen.png"
    if os.path.exists(screenshot_path):
        return FileResponse(screenshot_path)
    return JSONResponse(status_code=404, content={"error": "No screenshot available"})

@app.get("/is_active")
async def is_active(current_user: str = Depends(auth_manager.get_current_user)):
    return {"is_active": False}  # No active agent in this mock

@app.get("/stop")
async def stop(current_user: str = Depends(auth_manager.get_current_user)):
    return JSONResponse(status_code=200, content={"status": "stopped"})

@app.get("/latest_answer")
async def get_latest_answer(current_user: str = Depends(auth_manager.get_current_user)):
    global query_resp_history
    
    if query_resp_history:
        return JSONResponse(status_code=200, content=query_resp_history[-1])
    
    # Return empty response
    return JSONResponse(status_code=200, content={
        "done": "false",
        "answer": "",
        "reasoning": "",
        "agent_name": "Friday",
        "success": "true",
        "blocks": {},
        "status": "Ready",
        "uid": str(uuid.uuid4())
    })

@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest, 
    current_user: str = Depends(auth_manager.get_current_user)
):
    global is_generating, query_resp_history
    
    if is_generating:
        return JSONResponse(status_code=429, content={"error": "Another query is being processed"})
    
    try:
        is_generating = True
        
        # Simulate processing delay
        await asyncio.sleep(1)
        
        # Generate mock response based on query
        query_lower = request.query.lower()
        
        if "capital" in query_lower and "france" in query_lower:
            answer = mock_agent_responses["capital"]
            agent = "Browser"
        elif "2+2" in query_lower or "2 + 2" in query_lower:
            answer = mock_agent_responses["math"]
            agent = "Coder"
        elif "hello" in query_lower or "hi" in query_lower:
            answer = mock_agent_responses["hello"]
            agent = "Friday"
        else:
            answer = f"I received your query: '{request.query}'. In a full implementation, I would process this with the appropriate AI agent."
            agent = "Friday"
        
        # Create response
        response = QueryResponse(
            done="true",
            answer=answer,
            reasoning=f"Processed query for user {current_user}",
            agent_name=agent,
            success="true",
            blocks={
                "0": {
                    "tool_type": "mock",
                    "block": f"Query: {request.query}\nUser: {current_user}",
                    "feedback": "Mock response generated",
                    "success": True
                }
            },
            status="Completed",
            uid=str(uuid.uuid4())
        )
        
        # Store in history
        query_resp_history.append(response.jsonify())
        
        # Keep only last 10 responses
        if len(query_resp_history) > 10:
            query_resp_history = query_resp_history[-10:]
        
        return JSONResponse(status_code=200, content=response.jsonify())
        
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Error processing query: {str(e)}"}
        )
    finally:
        is_generating = False

# Create screenshots directory
if not os.path.exists(".screenshots"):
    os.makedirs(".screenshots")

if __name__ == "__main__":
    import uvicorn
    print("Starting AgenticSeek API with authentication...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)