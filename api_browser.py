#!/usr/bin/env python3
"""
Simplified API with browser support for AgenticSeek
"""

import os
import sys
import uuid
import configparser
import asyncio
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import google.generativeai as genai

# Import auth and schemas
from sources.auth import AuthManager, UserCredentials, TokenResponse
from sources.schemas import QueryRequest, QueryResponse
from sources.browser import Browser
from sources.logger import Logger

# Read config
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize
app = FastAPI(title="AgenticSeek Browser API", version="0.1.0")
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
api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyBj_AXSeaKBSazrWnx5NAsKm32k8sYjkxk")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

# Create screenshots directory
if not os.path.exists(".screenshots"):
    os.makedirs(".screenshots")
app.mount("/screenshots", StaticFiles(directory=".screenshots"), name="screenshots")

# Global variables
browser = None
is_generating = False
last_response = {}

def initialize_browser():
    """Initialize browser instance"""
    global browser
    try:
        browser = Browser(
            headless=config.getboolean('BROWSER', 'headless_browser'),
            stealth_mode=config.getboolean('BROWSER', 'stealth_mode', fallback=False)
        )
        logger.info("Browser initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize browser: {e}")
        return False

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize_browser()

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

@app.get("/latest_answer")
async def get_latest_answer(current_user: str = Depends(auth_manager.get_current_user)):
    global last_response
    if last_response:
        return JSONResponse(status_code=200, content=last_response)
    
    return JSONResponse(status_code=200, content={
        "done": "false",
        "answer": "",
        "reasoning": "",
        "agent_name": config["MAIN"]["agent_name"],
        "success": "true",
        "blocks": {},
        "status": "Ready",
        "uid": str(uuid.uuid4())
    })

async def browse_and_search(query: str):
    """Use browser to search for information"""
    global browser
    
    if not browser:
        return "Browser not initialized"
    
    try:
        # Navigate to Google
        browser.navigate("https://www.google.com")
        await asyncio.sleep(2)
        
        # Search
        browser.fill_input(query)
        browser.click_button("Search")
        await asyncio.sleep(3)
        
        # Take screenshot
        browser.screenshot()
        
        # Get page content
        content = browser.get_markdown()
        
        # Use Gemini to analyze
        prompt = f"Based on this search result, answer the user's query: {query}\n\nSearch results:\n{content[:2000]}"
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        logger.error(f"Browser error: {e}")
        return f"Error during web search: {str(e)}"

@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest, 
    current_user: str = Depends(auth_manager.get_current_user)
):
    global is_generating, last_response
    
    if is_generating:
        return JSONResponse(status_code=429, content={"error": "Another query is being processed"})
    
    try:
        is_generating = True
        query = request.query.lower()
        
        # Determine if we should use browser
        use_browser = any(keyword in query for keyword in [
            "search", "web", "browse", "find", "flight", "google", 
            "website", "online", "internet", "look up"
        ])
        
        if use_browser:
            # Use browser for web searches
            answer = await browse_and_search(request.query)
            agent = "Browser"
            blocks = {"browser": {"tool_type": "browser", "block": "Web search completed", "feedback": "Success", "success": True}}
        else:
            # Use Gemini directly for other queries
            response = model.generate_content(request.query)
            answer = response.text
            agent = "Friday"
            blocks = {}
        
        # Create response
        last_response = {
            "done": "true",
            "answer": answer,
            "reasoning": f"Processed using {agent} agent",
            "agent_name": agent,
            "success": "true",
            "blocks": blocks,
            "status": "Complete",
            "uid": str(uuid.uuid4())
        }
        
        return QueryResponse(**last_response)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
        
    finally:
        is_generating = False

@app.get("/stop")
async def stop(current_user: str = Depends(auth_manager.get_current_user)):
    global is_generating
    is_generating = False
    return JSONResponse(status_code=200, content={"status": "stopped"})

if __name__ == "__main__":
    print("Starting AgenticSeek API with Browser Support...")
    print("Browser will activate for web searches and flight queries")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)