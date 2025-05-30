#!/usr/bin/env python3
"""
AgenticSeek API with Google Gemini integration and authentication
"""

import os
import sys
import uuid
import asyncio
import configparser
from typing import Dict, List
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai

# Import our auth system
from sources.auth import AuthManager, UserCredentials, TokenResponse
from sources.schemas import QueryRequest, QueryResponse
from sources.logger import Logger

# Initialize
app = FastAPI(title="AgenticSeek API", version="0.1.0")
auth_manager = AuthManager()
logger = Logger("backend.log")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY not found in environment variables")
    sys.exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

# Read config
config = configparser.ConfigParser()
config.read('config.ini')
agent_name = config.get('MAIN', 'agent_name', fallback='Friday')
work_dir = config.get('MAIN', 'work_dir', fallback='/tmp')

# Global state
is_generating = False
query_resp_history = []

# Create directories
if not os.path.exists(".screenshots"):
    os.makedirs(".screenshots")
app.mount("/screenshots", StaticFiles(directory=".screenshots"), name="screenshots")

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
    screenshot_path = ".screenshots/updated_screen.png"
    if os.path.exists(screenshot_path):
        return FileResponse(screenshot_path)
    return JSONResponse(status_code=404, content={"error": "No screenshot available"})

@app.get("/is_active")
async def is_active(current_user: str = Depends(auth_manager.get_current_user)):
    return {"is_active": is_generating}

@app.get("/stop")
async def stop(current_user: str = Depends(auth_manager.get_current_user)):
    global is_generating
    is_generating = False
    return JSONResponse(status_code=200, content={"status": "stopped"})

@app.get("/latest_answer")
async def get_latest_answer(current_user: str = Depends(auth_manager.get_current_user)):
    global query_resp_history
    
    if query_resp_history:
        return JSONResponse(status_code=200, content=query_resp_history[-1])
    
    return JSONResponse(status_code=200, content={
        "done": "false",
        "answer": "",
        "reasoning": "",
        "agent_name": agent_name,
        "success": "true",
        "blocks": {},
        "status": "Ready",
        "uid": str(uuid.uuid4())
    })

async def process_with_gemini(query: str, user: str) -> Dict:
    """Process query with Google Gemini"""
    try:
        # Create a context-aware prompt
        prompt = f"""You are {agent_name}, an AI assistant helping user '{user}'.
        
User Query: {query}

Please provide a helpful, accurate, and friendly response. If the query involves:
- Code: Provide clear code examples with explanations
- Questions: Give comprehensive answers
- Tasks: Break down into clear steps

Response:"""
        
        # Generate response
        response = await asyncio.to_thread(model.generate_content, prompt)
        
        return {
            "answer": response.text,
            "reasoning": f"Processed query using Gemini for user {user}",
            "agent": agent_name,
            "success": True
        }
    except Exception as e:
        logger.error(f"Gemini error: {str(e)}")
        return {
            "answer": f"I encountered an error: {str(e)}",
            "reasoning": "Error during processing",
            "agent": agent_name,
            "success": False
        }

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
        logger.info(f"Processing query from {current_user}: {request.query}")
        
        # Process with Gemini
        result = await process_with_gemini(request.query, current_user)
        
        # Create response
        response = QueryResponse(
            done="true",
            answer=result["answer"],
            reasoning=result["reasoning"],
            agent_name=result["agent"],
            success=str(result["success"]).lower(),
            blocks={
                "0": {
                    "tool_type": "gemini",
                    "block": f"Query: {request.query}",
                    "feedback": "Response generated",
                    "success": result["success"]
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
        
        logger.info(f"Query processed successfully for {current_user}")
        return JSONResponse(status_code=200, content=response.jsonify())
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Error processing query: {str(e)}"}
        )
    finally:
        is_generating = False

if __name__ == "__main__":
    import uvicorn
    print("Starting AgenticSeek API with Google Gemini...")
    print(f"Using model: gemini-2.0-flash")
    print(f"Agent name: {agent_name}")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)