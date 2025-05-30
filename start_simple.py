#!/usr/bin/env python3
"""Simple startup script that avoids complex dependencies"""
import os
import sys

# Try to import and start the simple app first
try:
    import app
    # App will start itself
except Exception as e:
    print(f"Error starting app.py: {e}")
    
    # Fallback to minimal server
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI()
    
    @app.get("/")
    def root():
        return {"status": "Yappy is starting up..."}
    
    @app.get("/health")
    def health():
        return {"status": "healthy"}
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)