#!/usr/bin/env python3

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Create FastAPI app
app = FastAPI(title="Yappy AI API", version="0.1.0")

# Add CORS middleware
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
    model_name: Optional[str] = "openai"
    api_key: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    status: str = "success"

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

# Root endpoint
@app.get("/api")
async def root():
    return {"message": "Yappy AI API is running"}

# Simplified query endpoint
@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        if not request.api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        # For now, return a simple response
        # In production, this would call the actual LLM
        response = QueryResponse(
            answer=f"Woof! I'm Yappy! You asked: {request.query}. Please deploy the full version for complete functionality.",
            status="success"
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Yappy API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)