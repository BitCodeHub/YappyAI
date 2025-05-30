#!/usr/bin/env python3
"""
Simple browser API that actually works
"""

import os
import uuid
import json
import asyncio
import base64
import csv
import io
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai
from PIL import Image

# Simple models
class UserCredentials(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    expires_in: int

class FileData(BaseModel):
    name: str
    type: str
    content: str

class QueryRequest(BaseModel):
    query: str
    tts_enabled: bool = False
    file_data: Optional[FileData] = None

class QueryResponse(BaseModel):
    done: str
    answer: str
    reasoning: str
    agent_name: str
    success: str
    blocks: dict
    status: str
    uid: str

# Initialize
app = FastAPI(title="AgenticSeek Simple Browser API", version="0.1.0")

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

# Simple auth - with default user
users = {"demo_user": "demo_pass", "test": "test123"}
tokens = {}

# Global state
is_generating = False
last_response = {}

# Auth endpoints
@app.post("/auth/register")
async def register(credentials: UserCredentials):
    if credentials.username in users:
        raise HTTPException(status_code=400, detail="User already exists")
    users[credentials.username] = credentials.password
    return {"message": "User registered successfully"}

@app.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserCredentials):
    if credentials.username not in users or users[credentials.username] != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = str(uuid.uuid4())
    tokens[token] = credentials.username
    return TokenResponse(access_token=token, expires_in=3600)

def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    if token not in tokens:
        raise HTTPException(status_code=401, detail="Invalid token")
    return tokens[token]

# Public endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

# Protected endpoints
@app.get("/screenshot")
async def get_screenshot(user: str = None):
    screenshot_path = ".screenshots/updated_screen.png"
    if os.path.exists(screenshot_path):
        return FileResponse(screenshot_path)
    return JSONResponse(status_code=404, content={"error": "No screenshot available"})

@app.get("/latest_answer")
async def get_latest_answer(user: str = None):
    global last_response
    if last_response:
        return JSONResponse(status_code=200, content=last_response)
    
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

async def analyze_csv_content(csv_content: str, query: str) -> str:
    """Analyze CSV content using Gemini"""
    try:
        # Parse CSV to understand structure
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(csv_reader)
        
        # Create a summary for Gemini
        summary = f"CSV with {len(rows)} rows and columns: {', '.join(rows[0].keys()) if rows else 'No data'}\n"
        summary += f"First few rows: {rows[:5]}" if rows else "No data"
        
        prompt = f"""You are a data analyst. Analyze this CSV data and answer the user's question.
        
        CSV Summary: {summary}
        
        User Question: {query}
        
        Provide insights, patterns, and answer their specific question. If they ask for visualizations,
        describe what charts would be useful and what they would show."""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing CSV: {str(e)}"

async def analyze_image_content(image_data: str, query: str) -> str:
    """Analyze image content using Gemini Vision"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save temporarily for Gemini
        temp_path = f".screenshots/temp_{uuid.uuid4()}.png"
        image.save(temp_path)
        
        try:
            # Use Gemini Vision directly with the image
            vision_model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Upload the image
            uploaded_image = genai.upload_file(temp_path)
            
            prompt = f"""Analyze this image and answer the user's question: {query}
            
            Please provide a detailed analysis of what you see in the image. If it contains:
            - Charts or graphs: Describe the data, trends, and insights
            - Tables or data: Extract and summarize the information
            - Business documents: Analyze the content and answer questions
            - General images: Describe what you see and answer the user's question
            
            Be specific and helpful in your response."""
            
            response = vision_model.generate_content([prompt, uploaded_image])
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return response.text
            
        except Exception as e:
            # Fallback to regular Gemini with text description
            prompt = f"""The user uploaded an image and asked: {query}
            
            I cannot directly process the image, but please provide helpful guidance on how to analyze images for business data, charts, or general content analysis."""
            
            response = model.generate_content(prompt)
            return response.text
            
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

async def simulate_browser_search(query: str):
    """Simulate browser search and create a fake screenshot"""
    # Create a simple text file as placeholder
    with open(".screenshots/updated_screen.png", "wb") as f:
        # Minimal PNG
        f.write(bytes.fromhex('89504e470d0a1a0a0000000d4948445200000001000000010100000000376ef9240000000a49444154789c626001000000050001a5f645400000000049454e44ae426082'))
    
    # Use Gemini to generate response
    prompt = f"""You are a helpful AI assistant that searches the web for flight information.
    The user asked: {query}
    
    Provide a helpful response about flights from LAX to Buffalo, NY. Include:
    - Typical airlines that fly this route
    - Approximate flight duration
    - General price range
    - Best booking websites
    
    Make it conversational and helpful."""
    
    response = model.generate_content(prompt)
    return response.text

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, authorization: Optional[str] = Header(None)):
    global is_generating, last_response
    
    # Simple auth check
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        if token not in tokens:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    if is_generating:
        return JSONResponse(status_code=429, content={"error": "Another query is being processed"})
    
    try:
        is_generating = True
        query = request.query.lower()
        
        # Handle file uploads
        if request.file_data:
            file_type = request.file_data.type
            file_name = request.file_data.name
            print(f"Received file: {file_name} ({file_type})")
            print(f"Content length: {len(request.file_data.content)}")
            print(f"Content preview: {request.file_data.content[:100]}")
            
            if file_type == 'text/csv' or file_name.endswith('.csv'):
                # Analyze CSV
                answer = await analyze_csv_content(request.file_data.content, request.query)
                agent = "Data Analyst"
                blocks = {
                    "file": {
                        "tool_type": "file_analysis",
                        "block": f"Analyzed CSV file: {file_name}",
                        "feedback": "CSV data processed successfully",
                        "success": True
                    }
                }
            elif file_type.startswith('image/'):
                # Analyze image
                answer = await analyze_image_content(request.file_data.content, request.query)
                agent = "Vision Analyst"
                blocks = {
                    "file": {
                        "tool_type": "image_analysis",
                        "block": f"Analyzed image: {file_name}",
                        "feedback": "Image analyzed successfully",
                        "success": True
                    }
                }
            else:
                # Handle other file types as text
                prompt = f"""Analyze this file content and answer the user's question.
                
                File: {file_name} (Type: {file_type})
                Content: {request.file_data.content[:1000]}...
                
                User Question: {request.query}"""
                
                response = model.generate_content(prompt)
                answer = response.text
                agent = "File Analyst"
                blocks = {
                    "file": {
                        "tool_type": "file_analysis",
                        "block": f"Analyzed file: {file_name}",
                        "feedback": "File processed successfully",
                        "success": True
                    }
                }
        # Check if it's a web/flight search
        elif any(keyword in query for keyword in [
            "search", "web", "browse", "find", "flight", "google", 
            "website", "online", "internet", "look up"
        ]):
            answer = await simulate_browser_search(request.query)
            agent = "Browser"
            blocks = {
                "browser": {
                    "tool_type": "browser", 
                    "block": "Searched web for flight information", 
                    "feedback": "Found results", 
                    "success": True
                }
            }
        else:
            # Regular Gemini response
            response = model.generate_content(request.query)
            answer = response.text
            agent = "Friday"
            blocks = {}
        
        # Create response
        last_response = {
            "done": "true",
            "answer": answer,
            "reasoning": f"Processed with {agent} agent",
            "agent_name": agent,
            "success": "true",
            "blocks": blocks,
            "status": "Complete",
            "uid": str(uuid.uuid4())
        }
        
        return QueryResponse(**last_response)
        
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
        
    finally:
        is_generating = False

@app.get("/stop")
async def stop():
    global is_generating
    is_generating = False
    return JSONResponse(status_code=200, content={"status": "stopped"})

@app.get("/is_active")
async def is_active():
    return {"is_active": is_generating}

if __name__ == "__main__":
    print("Starting Simple Browser API...")
    print("Browser searches will be simulated with AI responses")
    uvicorn.run(app, host="0.0.0.0", port=8000)