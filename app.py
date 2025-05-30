import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Yappy AI", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    timestamp: str

@app.get("/")
def read_root():
    return {
        "message": "Woof! 🐕 Welcome to Yappy AI!",
        "status": "ready",
        "endpoints": {
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {
        "version": "1.0.2", 
        "update": "Yappy welcome message",
        "deployed_at": datetime.now().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with Yappy AI"""
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # For now, echo back with Yappy personality
    response_text = f"Woof! 🐕 You said: '{request.message}'. I'm Yappy, your friendly AI assistant! Once we connect AI providers, I'll be even more helpful!"
    
    return ChatResponse(
        response=response_text,
        timestamp=datetime.now().isoformat()
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting on port {port}", flush=True)
    print(f"PORT env var: {os.environ.get('PORT', 'not set')}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)