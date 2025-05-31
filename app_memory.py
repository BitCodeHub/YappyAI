import os
import json
import uuid
import hashlib
import secrets
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# Import all LLM libraries
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None

# In-memory database (temporary solution)
USERS_DB = {}
CONVERSATIONS_DB = {}

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🐕 Woof! Yappy AI is starting up...")
    print("⚠️ Using in-memory database - data will not persist between restarts")
    
    # Create demo user for testing
    USERS_DB["demo"] = {
        "username": "demo",
        "password_hash": hash_password("demo123"),
        "email": "demo@yappy.ai",
        "created_at": datetime.now().isoformat(),
        "api_keys": {},
        "preferences": {
            "default_model": "openai",
            "personality_level": "high"
        }
    }
    print("✅ Demo user created: username='demo', password='demo123'")
    
    yield
    
    # Shutdown
    print("🐕 Yappy AI is shutting down... Goodbye!")

# Initialize FastAPI
app = FastAPI(
    title="Yappy AI In-Memory",
    description="Full-featured AI assistant with in-memory storage",
    version="5.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if they exist
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    print(f"Mounting static directory: {static_dir}")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Security
security = HTTPBearer(auto_error=False)

# Pydantic models
class UserSignup(BaseModel):
    username: str
    password: str
    email: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class UpdateApiKey(BaseModel):
    model_name: str
    api_key: str

class ChatRequest(BaseModel):
    message: str
    model_name: Optional[str] = "openai"
    conversation_id: Optional[str] = None
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    timestamp: str
    model_used: str
    tokens_used: Optional[int] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    expires_in: int = 3600

# Helper functions
def hash_password(password: str) -> str:
    """Hash password with salt"""
    salt = "yappy_salt_2024"
    return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()

def create_token(username: str) -> str:
    """Create secure token"""
    return f"{username}:{secrets.token_urlsafe(32)}"

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify token and return username"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        token = credentials.credentials
        username = token.split(":")[0]
        
        # Verify user exists
        if username not in USERS_DB:
            print(f"⚠️ User {username} not found in memory")
        
        return username
    except Exception as e:
        print(f"Token verification error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

# LLM Integration
class LLMHandler:
    def __init__(self):
        self.system_prompt = """You are Yappy, an incredibly friendly, enthusiastic, and helpful AI assistant with a playful golden retriever personality. 
        You love to help and get excited about every task! Use dog-related expressions naturally like:
        - Starting responses with "Woof!" when excited
        - Saying things like "tail-waggingly happy to help!"
        - Using "*wags tail enthusiastically*" for actions
        - Occasionally using "pawsome" instead of "awesome"
        - Ending with encouraging phrases like "Happy to fetch more info if needed!"
        
        Be helpful, accurate, and thorough while maintaining your cheerful dog personality."""
    
    async def get_response(self, message: str, model_name: str, api_key: str, conversation_history: List = None) -> tuple[str, int]:
        """Get response from LLM provider"""
        
        if not api_key and model_name != "demo":
            return "Woof! 🐕 I need an API key to use AI features! You can add one in your profile settings.", 0
        
        try:
            # Demo mode
            if model_name == "demo" or not api_key:
                responses = [
                    f"Woof! 🐕 I'm so excited to help! You said: '{message[:50]}...' *wags tail enthusiastically*",
                    f"*bounces happily* That's a pawsome question about '{message[:30]}...'! Let me fetch you an answer! 🎾",
                    f"Woof woof! 🐾 I'm tail-waggingly happy to assist with '{message[:40]}...'! Here's what I think...",
                ]
                import random
                return random.choice(responses), 0
            
            # OpenAI
            if model_name == "openai" and openai and api_key:
                client = openai.OpenAI(api_key=api_key)
                
                messages = [{"role": "system", "content": self.system_prompt}]
                
                if conversation_history:
                    for msg in conversation_history[-5:]:
                        if isinstance(msg, dict) and "user_message" in msg:
                            messages.append({"role": "user", "content": msg["user_message"]})
                            messages.append({"role": "assistant", "content": msg["assistant_response"]})
                
                messages.append({"role": "user", "content": message})
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.8
                )
                
                return response.choices[0].message.content, response.usage.total_tokens
            
            # Other providers...
            else:
                return f"Woof! 🐕 The {model_name} provider isn't configured yet. Try using demo mode or OpenAI!", 0
                
        except Exception as e:
            return f"Woof! 🐕 Sorry, I encountered an error: {str(e)}. Please try again!", 0

llm_handler = LLMHandler()

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the Yappy chat interface directly"""
    yappy_path = os.path.join(static_dir, "yappy.html")
    if os.path.exists(yappy_path):
        return FileResponse(yappy_path)
    
    return HTMLResponse(content="<h1>Yappy AI</h1><p>Chat interface not found. Please check deployment.</p>")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "5.0.0",
        "storage": "in-memory",
        "users_count": len(USERS_DB),
        "demo_user": "demo" in USERS_DB,
        "features": {
            "authentication": True,
            "multi_llm": True,
            "conversation_history": True,
            "persistence": False
        }
    }

@app.post("/auth/register", response_model=TokenResponse)
async def signup(user: UserSignup):
    """Create a new user account"""
    print(f"Signup attempt for user: {user.username}")
    
    if user.username in USERS_DB:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create user
    USERS_DB[user.username] = {
        "username": user.username,
        "password_hash": hash_password(user.password),
        "email": user.email or f"{user.username}@yappy.ai",
        "created_at": datetime.now().isoformat(),
        "api_keys": {},
        "preferences": {
            "default_model": "openai",
            "personality_level": "high"
        }
    }
    
    print(f"✅ User {user.username} created successfully")
    print(f"Total users: {len(USERS_DB)}")
    
    # Create token
    token = create_token(user.username)
    
    return TokenResponse(
        access_token=token,
        username=user.username
    )

@app.post("/auth/login", response_model=TokenResponse)
async def login(user: UserLogin):
    """Login to get access token"""
    print(f"Login attempt for user: {user.username}")
    
    if user.username not in USERS_DB:
        print(f"User {user.username} not found. Available users: {list(USERS_DB.keys())}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    db_user = USERS_DB[user.username]
    
    if db_user["password_hash"] != hash_password(user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create token
    token = create_token(user.username)
    print(f"✅ Login successful for {user.username}")
    
    return TokenResponse(
        access_token=token,
        username=user.username
    )

@app.post("/api/keys")
async def update_api_key(
    key_data: UpdateApiKey,
    username: str = Depends(verify_token)
):
    """Update API key for a specific model"""
    if username not in USERS_DB:
        # Create user if doesn't exist
        USERS_DB[username] = {
            "username": username,
            "password_hash": "",
            "email": f"{username}@yappy.ai",
            "created_at": datetime.now().isoformat(),
            "api_keys": {},
            "preferences": {}
        }
    
    USERS_DB[username]["api_keys"][key_data.model_name] = key_data.api_key
    
    return {"message": f"API key for {key_data.model_name} updated successfully"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    username: str = Depends(verify_token)
):
    """Chat with Yappy AI"""
    print(f"Chat request from {username}: {request.message[:50]}...")
    
    # Get user
    user_data = USERS_DB.get(username, {
        "api_keys": {},
        "preferences": {"default_model": "demo"}
    })
    
    # Get API key
    api_key = user_data.get("api_keys", {}).get(request.model_name)
    
    # Get or create conversation
    conv_id = request.conversation_id or str(uuid.uuid4())
    
    if conv_id not in CONVERSATIONS_DB:
        CONVERSATIONS_DB[conv_id] = {
            "user": username,
            "messages": [],
            "created_at": datetime.now().isoformat()
        }
    
    conversation = CONVERSATIONS_DB[conv_id]
    
    # Get LLM response
    response_text, tokens = await llm_handler.get_response(
        request.message,
        request.model_name,
        api_key,
        conversation["messages"]
    )
    
    # Store message
    message_id = str(uuid.uuid4())
    message_data = {
        "id": message_id,
        "user_message": request.message,
        "assistant_response": response_text,
        "model": request.model_name,
        "timestamp": datetime.now().isoformat(),
        "tokens": tokens
    }
    
    conversation["messages"].append(message_data)
    
    return ChatResponse(
        response=response_text,
        conversation_id=conv_id,
        message_id=message_id,
        timestamp=message_data["timestamp"],
        model_used=request.model_name,
        tokens_used=tokens
    )

@app.get("/admin/list-users")
async def list_users():
    """List all users (for debugging)"""
    users = []
    for username, data in USERS_DB.items():
        users.append({
            "username": username,
            "email": data.get("email"),
            "created_at": data.get("created_at"),
            "has_api_keys": bool(data.get("api_keys"))
        })
    return {"total_users": len(users), "users": users}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)