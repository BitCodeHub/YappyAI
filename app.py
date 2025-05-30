import os
from datetime import datetime
from fastapi import FastAPI
import uvicorn

app = FastAPI()

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting on port {port}", flush=True)
    print(f"PORT env var: {os.environ.get('PORT', 'not set')}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)