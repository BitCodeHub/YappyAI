#!/usr/bin/env python3

import os, sys
import uvicorn
import aiofiles
import configparser
import asyncio
import time
from typing import List
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uuid

from sources.llm_provider import Provider
from sources.interaction import Interaction
from sources.agents import CasualAgent, CoderAgent, FileAgent, PlannerAgent, BrowserAgent
from sources.browser import Browser, create_driver
from sources.utility import pretty_print
from sources.logger import Logger
from sources.schemas import QueryRequest, QueryResponse
from sources.auth import AuthManager, UserCredentials, TokenResponse
from sources.feature_flags import feature_flags


from celery import Celery

api = FastAPI(title="AgenticSeek API", version="0.1.0")
celery_app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")
celery_app.conf.update(task_track_started=True)
logger = Logger("backend.log")
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize authentication manager
auth_manager = AuthManager()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists(".screenshots"):
    os.makedirs(".screenshots")
api.mount("/screenshots", StaticFiles(directory=".screenshots"), name="screenshots")

# Serve frontend static files if they exist
if os.path.exists("static"):
    api.mount("/", StaticFiles(directory="static", html=True), name="static")

def get_default_model_for_provider(provider_name):
    """Get default model for each provider"""
    model_mapping = {
        "openai": "gpt-4",
        "anthropic": "claude-3-opus-20240229",
        "google": "gemini-1.5-pro",
        "groq": "llama3-70b-8192",
        "ollama": "llama3"
    }
    return model_mapping.get(provider_name, "gpt-4")

def create_interaction_with_provider(provider):
    """Create a new interaction with a specific provider"""
    stealth_mode = config.getboolean('BROWSER', 'stealth_mode')
    personality_folder = "jarvis" if config.getboolean('MAIN', 'jarvis_personality') else "base"
    languages = config["MAIN"]["languages"].split(' ')
    
    browser = Browser(
        create_driver(headless=config.getboolean('BROWSER', 'headless_browser'), stealth_mode=stealth_mode, lang=languages[0]),
        anticaptcha_manual_install=stealth_mode
    )
    
    casual_agent = CasualAgent(provider, f"prompts/{personality_folder}/casual_agent.txt")
    coder_agent = CoderAgent(provider, f"prompts/{personality_folder}/coder_agent.txt")
    file_agent = FileAgent(provider, f"prompts/{personality_folder}/file_agent.txt")
    planner_agent = PlannerAgent(provider, f"prompts/{personality_folder}/planner_agent.txt")
    browser_agent = BrowserAgent(provider, f"prompts/{personality_folder}/browser_agent.txt", browser)

    agents = [casual_agent, coder_agent, file_agent, planner_agent, browser_agent]
    
    return Interaction(
        agents, 
        provider,
        is_voice_enabled=config.getboolean('VOICE', 'is_voice_enabled'),
        is_tts_enabled=config.getboolean('VOICE', 'is_tts_enabled')
    )

def initialize_system():
    stealth_mode = config.getboolean('BROWSER', 'stealth_mode')
    personality_folder = "jarvis" if config.getboolean('MAIN', 'jarvis_personality') else "base"
    languages = config["MAIN"]["languages"].split(' ')

    provider = Provider(
        provider_name=config["MAIN"]["provider_name"],
        model=config["MAIN"]["provider_model"],
        server_address=config["MAIN"]["provider_server_address"],
        is_local=config.getboolean('MAIN', 'is_local')
    )
    logger.info(f"Provider initialized: {provider.provider_name} ({provider.model})")

    browser = Browser(
        create_driver(headless=config.getboolean('BROWSER', 'headless_browser'), stealth_mode=stealth_mode, lang=languages[0]),
        anticaptcha_manual_install=stealth_mode
    )
    logger.info("Browser initialized")

    agents = [
        CasualAgent(
            name=config["MAIN"]["agent_name"],
            prompt_path=f"prompts/{personality_folder}/casual_agent.txt",
            provider=provider, verbose=False
        ),
        CoderAgent(
            name="coder",
            prompt_path=f"prompts/{personality_folder}/coder_agent.txt",
            provider=provider, verbose=False
        ),
        FileAgent(
            name="File Agent",
            prompt_path=f"prompts/{personality_folder}/file_agent.txt",
            provider=provider, verbose=False
        ),
        BrowserAgent(
            name="Browser",
            prompt_path=f"prompts/{personality_folder}/browser_agent.txt",
            provider=provider, verbose=False, browser=browser
        ),
        PlannerAgent(
            name="Planner",
            prompt_path=f"prompts/{personality_folder}/planner_agent.txt",
            provider=provider, verbose=False, browser=browser
        )
    ]
    logger.info("Agents initialized")

    interaction = Interaction(
        agents,
        tts_enabled=config.getboolean('MAIN', 'speak'),
        stt_enabled=config.getboolean('MAIN', 'listen'),
        recover_last_session=config.getboolean('MAIN', 'recover_last_session'),
        langs=languages
    )
    logger.info("Interaction initialized")
    return interaction

interaction = initialize_system()
is_generating = False
query_resp_history = []

# Authentication endpoints
@api.post("/auth/register")
async def register(credentials: UserCredentials):
    """Register a new user"""
    try:
        user = auth_manager.register_user(credentials.username, credentials.password)
        logger.info(f"User registered: {user.username}")
        return user
    except HTTPException as e:
        logger.error(f"Registration failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@api.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserCredentials):
    """Login and receive access token"""
    username = auth_manager.authenticate_user(credentials.username, credentials.password)
    if not username:
        logger.warning(f"Login failed for user: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth_manager.create_access_token(username)
    logger.info(f"User logged in: {username}")
    
    return TokenResponse(
        access_token=access_token,
        expires_in=auth_manager.token_expiry_hours * 3600
    )

@api.get("/auth/me")
async def get_current_user(current_user: str = Depends(auth_manager.get_current_user)):
    """Get current authenticated user"""
    return {"username": current_user}

@api.get("/screenshot")
async def get_screenshot(current_user: str = Depends(auth_manager.get_current_user)):
    logger.info(f"Screenshot endpoint called by user: {current_user}")
    screenshot_path = ".screenshots/updated_screen.png"
    if os.path.exists(screenshot_path):
        return FileResponse(screenshot_path)
    logger.error("No screenshot available")
    return JSONResponse(
        status_code=404,
        content={"error": "No screenshot available"}
    )

@api.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy", "version": "0.1.0"}

@api.get("/is_active")
async def is_active(current_user: str = Depends(auth_manager.get_current_user)):
    logger.info(f"Is active endpoint called by user: {current_user}")
    return {"is_active": interaction.is_active}

@api.get("/stop")
async def stop(current_user: str = Depends(auth_manager.get_current_user)):
    logger.info(f"Stop endpoint called by user: {current_user}")
    interaction.current_agent.request_stop()
    return JSONResponse(status_code=200, content={"status": "stopped"})

@api.get("/latest_answer")
async def get_latest_answer(current_user: str = Depends(auth_manager.get_current_user)):
    global query_resp_history
    if interaction.current_agent is None:
        return JSONResponse(status_code=404, content={"error": "No agent available"})
    uid = str(uuid.uuid4())
    if not any(q["answer"] == interaction.current_agent.last_answer for q in query_resp_history):
        query_resp = {
            "done": "false",
            "answer": interaction.current_agent.last_answer,
            "reasoning": interaction.current_agent.last_reasoning,
            "agent_name": interaction.current_agent.agent_name if interaction.current_agent else "None",
            "success": interaction.current_agent.success,
            "blocks": {f'{i}': block.jsonify() for i, block in enumerate(interaction.get_last_blocks_result())} if interaction.current_agent else {},
            "status": interaction.current_agent.get_status_message if interaction.current_agent else "No status available",
            "uid": uid
        }
        interaction.current_agent.last_answer = ""
        interaction.current_agent.last_reasoning = ""
        query_resp_history.append(query_resp)
        return JSONResponse(status_code=200, content=query_resp)
    if query_resp_history:
        return JSONResponse(status_code=200, content=query_resp_history[-1])
    return JSONResponse(status_code=404, content={"error": "No answer available"})

async def think_wrapper(interaction, query):
    try:
        interaction.last_query = query
        logger.info("Agents request is being processed")
        success = await interaction.think()
        if not success:
            interaction.last_answer = "Error: No answer from agent"
            interaction.last_reasoning = "Error: No reasoning from agent"
            interaction.last_success = False
        else:
            interaction.last_success = True
        pretty_print(interaction.last_answer)
        interaction.speak_answer()
        return success
    except Exception as e:
        logger.error(f"Error in think_wrapper: {str(e)}")
        interaction.last_answer = f""
        interaction.last_reasoning = f"Error: {str(e)}"
        interaction.last_success = False
        raise e

@api.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, current_user: str = Depends(auth_manager.get_current_user)):
    global is_generating, query_resp_history
    logger.info(f"Processing query: {request.query}")
    query_resp = QueryResponse(
        done="false",
        answer="",
        reasoning="",
        agent_name="Unknown",
        success="false",
        blocks={},
        status="Ready",
        uid=str(uuid.uuid4())
    )
    if is_generating:
        logger.warning("Another query is being processed, please wait.")
        return JSONResponse(status_code=429, content=query_resp.jsonify())

    try:
        is_generating = True
        
        # Create a new provider with user's LLM settings if provided
        current_interaction = interaction
        if request.llm_model and request.api_key:
            user_provider = Provider(
                provider_name=request.llm_model,
                model=get_default_model_for_provider(request.llm_model),
                api_key=request.api_key,
                server_address=config["MAIN"]["provider_server_address"] if request.llm_model == "ollama" else None,
                is_local=request.llm_model == "ollama"
            )
            # Create a new interaction with user's provider
            current_interaction = create_interaction_with_provider(user_provider)
            
        # Handle file data if provided
        if request.file_data:
            current_interaction.file_data = request.file_data
            
        success = await think_wrapper(current_interaction, request.query)
        is_generating = False

        if not success:
            query_resp.answer = current_interaction.last_answer
            query_resp.reasoning = current_interaction.last_reasoning
            return JSONResponse(status_code=400, content=query_resp.jsonify())

        if current_interaction.current_agent:
            blocks_json = {f'{i}': block.jsonify() for i, block in enumerate(current_interaction.current_agent.get_blocks_result())}
        else:
            logger.error("No current agent found")
            blocks_json = {}
            query_resp.answer = "Error: No current agent"
            return JSONResponse(status_code=400, content=query_resp.jsonify())

        logger.info(f"Answer: {current_interaction.last_answer}")
        logger.info(f"Blocks: {blocks_json}")
        query_resp.done = "true"
        query_resp.answer = current_interaction.last_answer
        query_resp.reasoning = current_interaction.last_reasoning
        query_resp.agent_name = current_interaction.current_agent.agent_name
        query_resp.success = str(current_interaction.last_success)
        query_resp.blocks = blocks_json
        
        query_resp_dict = {
            "done": query_resp.done,
            "answer": query_resp.answer,
            "agent_name": query_resp.agent_name,
            "success": query_resp.success,
            "blocks": query_resp.blocks,
            "status": query_resp.status,
            "uid": query_resp.uid
        }
        query_resp_history.append(query_resp_dict)

        logger.info("Query processed successfully")
        return JSONResponse(status_code=200, content=query_resp.jsonify())
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("Processing finished")
        if config.getboolean('MAIN', 'save_session'):
            interaction.save_session()

@api.get("/feature-flags")
async def get_feature_flags(user_id: str = None):
    """Get enabled feature flags for a user."""
    try:
        enabled_flags = feature_flags.get_enabled_flags(user_id)
        return JSONResponse(status_code=200, content={"flags": enabled_flags})
    except Exception as e:
        logger.error(f"Error getting feature flags: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

@api.post("/feature-flags/update")
async def update_feature_flag(flag_data: dict):
    """Update a feature flag (admin only)."""
    try:
        flag_name = flag_data.get("flag_name")
        enabled = flag_data.get("enabled")
        rollout_percentage = flag_data.get("rollout_percentage")
        allowed_users = flag_data.get("allowed_users")
        
        if not flag_name:
            return JSONResponse(status_code=400, content={"error": "flag_name is required"})
        
        feature_flags.update_flag(
            flag_name=flag_name,
            enabled=enabled,
            rollout_percentage=rollout_percentage,
            allowed_users=allowed_users
        )
        
        return JSONResponse(status_code=200, content={"message": "Feature flag updated successfully"})
    except Exception as e:
        logger.error(f"Error updating feature flag: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(api, host="0.0.0.0", port=port)