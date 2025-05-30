#!/usr/bin/env python3
"""
Full AgenticSeek API with authentication and all agent functionality
This combines the auth from api_working.py with the agents from api.py
"""

import os
import sys
import uuid
import json
import configparser
from typing import Dict
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import auth and schemas
from sources.auth import AuthManager, UserCredentials, TokenResponse
from sources.schemas import QueryRequest, QueryResponse

# Import the agent system
from sources.llm_provider import Provider
from sources.browser import Browser
from sources.interaction import Interaction
from sources.agents.planner_agent import PlannerAgent
from sources.agents.browser_agent import BrowserAgent
from sources.agents.code_agent import CoderAgent
from sources.agents.file_agent import FileAgent
from sources.agents.casual_agent import CasualAgent
from sources.router import AgentRouter
from sources.logger import Logger

# Read config
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize
app = FastAPI(title="AgenticSeek Full API", version="0.1.0")
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

# Create screenshots directory
if not os.path.exists(".screenshots"):
    os.makedirs(".screenshots")
app.mount("/screenshots", StaticFiles(directory=".screenshots"), name="screenshots")

# Global variables
interaction = None
is_generating = False
query_resp_history = []

def initialize_system():
    """Initialize the full agent system"""
    global interaction
    
    try:
        # Get configuration
        personality_folder = "jarvis" if config.getboolean('MAIN', 'jarvis_personality') else "base"
        languages = config["MAIN"]["languages"].split(' ')
        
        # Initialize provider with Google
        provider = Provider(
            provider_name="google",
            model="gemini-2.0-flash",
            is_local=False
        )
        
        # Initialize browser
        browser = Browser(
            headless=config.getboolean('BROWSER', 'headless_browser'),
            stealth_mode=config.getboolean('BROWSER', 'stealth_mode', fallback=False)
        )
        
        # Initialize agents
        agents = []
        
        # Planner agent
        with open(f"prompts/{personality_folder}/planner_agent.txt", "r") as file:
            prompt = file.read()
        agents.append(PlannerAgent(prompt=prompt, provider=provider))
        
        # Browser agent  
        with open(f"prompts/{personality_folder}/browser_agent.txt", "r") as file:
            prompt = file.read()
        agents.append(BrowserAgent(prompt=prompt, provider=provider, browser=browser))
        
        # Code agent
        with open(f"prompts/{personality_folder}/coder_agent.txt", "r") as file:
            prompt = file.read()
        agents.append(CoderAgent(prompt=prompt, provider=provider))
        
        # File agent
        with open(f"prompts/{personality_folder}/file_agent.txt", "r") as file:
            prompt = file.read()
        agents.append(FileAgent(prompt=prompt, provider=provider))
        
        # Casual agent
        with open(f"prompts/{personality_folder}/casual_agent.txt", "r") as file:
            prompt = file.read()
        agents.append(CasualAgent(prompt=prompt, provider=provider))
        
        # Initialize router
        router = AgentRouter(agents=agents, supported_language=languages)
        
        # Initialize interaction
        interaction = Interaction(
            router=router,
            provider=provider,
            agent_name=config["MAIN"]["agent_name"],
            work_dir=config["MAIN"]["work_dir"],
            personality_folder=personality_folder
        )
        
        logger.info("System initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        return False

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize_system()

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
    if interaction:
        interaction.force_stop = True
    return JSONResponse(status_code=200, content={"status": "stopped"})

@app.get("/latest_answer")
async def get_latest_answer(current_user: str = Depends(auth_manager.get_current_user)):
    if interaction and hasattr(interaction, 'last_response'):
        response = interaction.last_response
        return JSONResponse(status_code=200, content=response)
    
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

@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest, 
    current_user: str = Depends(auth_manager.get_current_user)
):
    global is_generating, interaction
    
    if is_generating:
        return JSONResponse(status_code=429, content={"error": "Another query is being processed"})
    
    if not interaction:
        return JSONResponse(status_code=500, content={"error": "System not initialized"})
    
    try:
        is_generating = True
        
        # Process query with full agent system
        # Run in thread since it might not be async
        import asyncio
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            response = await asyncio.get_event_loop().run_in_executor(
                executor, 
                interaction.interact, 
                request.query
            )
        
        # Parse response - interaction.interact returns formatted response
        if isinstance(response, dict):
            result = response
        else:
            # If it's a string, create basic response
            result = {
                "done": "true",
                "answer": str(response),
                "reasoning": "",
                "agent_name": config["MAIN"]["agent_name"],
                "success": "true",
                "blocks": {},
                "status": "Complete",
                "uid": str(uuid.uuid4())
            }
        
        # Store response
        interaction.last_response = result
        
        return QueryResponse(**interaction.last_response)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
        
    finally:
        is_generating = False

if __name__ == "__main__":
    print("Starting Full AgenticSeek API with all agents...")
    print("This includes Browser, Code, File, and Casual agents")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)