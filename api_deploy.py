#!/usr/bin/env python3

import os, sys
import uvicorn
import configparser
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid

# Conditional imports to handle missing dependencies gracefully
try:
    from sources.llm_provider import Provider
    from sources.logger import Logger
    logger = Logger("backend.log")
except:
    print("Warning: Some modules not available, running in limited mode")
    logger = None

# Create FastAPI app
app = FastAPI(title="Yappy AI", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files if they exist
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    model_name: Optional[str] = "openai"
    api_key: Optional[str] = None

class QueryResponse(BaseModel):
    uid: str
    query: str
    answer: str
    model_used: str
    status: str = "completed"

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

# Root API endpoint
@app.get("/api")
async def root():
    return {"message": "Yappy AI API is running", "version": "0.1.0"}

# Main query endpoint
@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        if not request.api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        # For deployment testing, return a simple response
        # In production, this would call the actual LLM
        uid = str(uuid.uuid4())
        response = QueryResponse(
            uid=uid,
            query=request.query,
            answer=f"Woof! I'm Yappy! You asked: '{request.query}'. The full version with {request.model_name} integration is being deployed!",
            model_used=request.model_name or "demo",
            status="completed"
        )
        
        if logger:
            logger.info(f"Query processed: {uid}")
            
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        if logger:
            logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Simple auth endpoints for frontend compatibility
@app.post("/auth/login")
async def login():
    # Simplified for deployment
    return {"access_token": "demo_token", "expires_in": 3600}

@app.get("/auth/me")
async def get_me():
    return {"username": "demo_user"}

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Yappy AI on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)