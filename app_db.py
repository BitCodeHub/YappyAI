import os
import json
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from databases import Database
import sqlalchemy
from sqlalchemy import MetaData, Table, Column, Integer, String, DateTime, Text, JSON

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

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./yappy.db")

# Fix for Render PostgreSQL URLs (they use postgres:// but we need postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

database = Database(DATABASE_URL)
metadata = MetaData()

# Database tables
users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String, unique=True, index=True),
    Column("email", String),
    Column("password_hash", String),
    Column("api_keys", JSON),
    Column("preferences", JSON),
    Column("created_at", DateTime, default=datetime.utcnow),
)

conversations_table = Table(
    "conversations",
    metadata,
    Column("id", String, primary_key=True),
    Column("user_id", Integer),
    Column("title", String),
    Column("messages", JSON),
    Column("created_at", DateTime, default=datetime.utcnow),
)

engine = sqlalchemy.create_engine(DATABASE_URL)

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Connect to database
    print("🐕 Woof! Yappy AI is starting up...")
    print(f"Database URL: {DATABASE_URL[:20]}...")
    
    try:
        await database.connect()
        print("✅ Database connected successfully")
        
        # Create tables if they don't exist
        metadata.create_all(bind=engine)
        print("✅ Database tables created/verified")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("⚠️ Will continue with limited functionality")
    
    yield
    
    # Shutdown
    try:
        await database.disconnect()
        print("🐕 Yappy AI is shutting down... Goodbye!")
    except:
        pass

# Initialize FastAPI
app = FastAPI(
    title="Yappy AI Complete with Database",
    description="Full-featured AI assistant with PostgreSQL database",
    version="4.0.0",
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
    print(f"Static files: {os.listdir(static_dir)}")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    print(f"Warning: Static directory not found at {static_dir}")

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
    salt = "yappy_salt_2024"  # In production, use unique salt per user
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
        
        # Verify user exists in database
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return username
    except:
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
            # OpenAI
            if model_name == "openai" and openai and api_key:
                client = openai.OpenAI(api_key=api_key)
                
                messages = [{"role": "system", "content": self.system_prompt}]
                
                # Add conversation history
                if conversation_history:
                    for msg in conversation_history[-5:]:  # Last 5 messages
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
            
            # Anthropic
            elif model_name == "anthropic" and anthropic and api_key:
                client = anthropic.Anthropic(api_key=api_key)
                
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=800,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": message}]
                )
                
                return response.content[0].text, response.usage.input_tokens + response.usage.output_tokens
            
            # Google Gemini
            elif model_name == "google" and genai and api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
                
                prompt = f"{self.system_prompt}\n\nUser: {message}"
                response = model.generate_content(prompt)
                
                return response.text, 100  # Gemini doesn't provide token count
            
            # Groq
            elif model_name == "groq" and Groq and api_key:
                client = Groq(api_key=api_key)
                
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message}
                ]
                
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.8
                )
                
                return response.choices[0].message.content, response.usage.total_tokens
            
            # Demo mode
            elif model_name == "demo":
                return f"Woof! 🐕 Demo response: I'm Yappy, your friendly AI assistant! You said: '{message[:50]}...' *wags tail happily*", 0
            
            else:
                return f"Woof! 🐕 Sorry, the {model_name} model isn't available right now. Please check your API key or try a different model!", 0
                
        except Exception as e:
            return f"Woof! 🐕 Sorry, I encountered an error: {str(e)}. Please try again or check your API key!", 0

llm_handler = LLMHandler()

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the Yappy chat interface directly"""
    yappy_path = os.path.join(static_dir, "yappy.html")
    if os.path.exists(yappy_path):
        print(f"Serving yappy.html from {yappy_path}")
        return FileResponse(yappy_path)
    
    # Fallback to the landing page if yappy.html doesn't exist
    print(f"yappy.html not found at {yappy_path}, serving landing page")
    return await root_landing()

@app.get("/landing", response_class=HTMLResponse)
async def root_landing():
    """Serve the main application"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Yappy AI - Database Edition</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            .container { max-width: 800px; margin: 0 auto; padding: 2rem; text-align: center; }
            h1 { font-size: 3rem; margin-bottom: 1rem; }
            .subtitle { font-size: 1.2rem; margin-bottom: 2rem; opacity: 0.9; }
            .btn { display: inline-block; padding: 12px 24px; background: rgba(255,255,255,0.2); color: white; text-decoration: none; border-radius: 8px; margin: 0 10px; transition: all 0.3s; }
            .btn:hover { background: rgba(255,255,255,0.3); transform: translateY(-2px); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐕 Yappy AI</h1>
            <p class="subtitle">Your friendly AI assistant with database persistence!</p>
            <a href="/docs" class="btn">API Documentation</a>
            <a href="/static/yappy.html" class="btn">Open Chat</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/chat")
async def chat_redirect():
    """Redirect to Yappy chat interface"""
    return RedirectResponse(url="/static/yappy.html", status_code=303)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0",
        "database": "connected" if database.is_connected else "disconnected",
        "features": {
            "authentication": True,
            "multi_llm": True,
            "conversation_history": True,
            "database_persistence": True,
        }
    }

@app.post("/auth/register", response_model=TokenResponse)
async def signup(user: UserSignup):
    """Create a new user account"""
    try:
        # Check if user already exists
        query = users_table.select().where(users_table.c.username == user.username)
        existing_user = await database.fetch_one(query)
        
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Create user
        user_data = {
            "username": user.username,
            "email": user.email or f"{user.username}@yappy.ai",
            "password_hash": hash_password(user.password),
            "api_keys": {},
            "preferences": {
                "default_model": "openai",
                "personality_level": "high"
            }
        }
        
        query = users_table.insert().values(**user_data)
        await database.execute(query)
        
        # Create token
        token = create_token(user.username)
        
        return TokenResponse(
            access_token=token,
            username=user.username
        )
    except Exception as e:
        print(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")

@app.post("/auth/login", response_model=TokenResponse)
async def login(user: UserLogin):
    """Login to get access token"""
    try:
        query = users_table.select().where(users_table.c.username == user.username)
        db_user = await database.fetch_one(query)
        
        if not db_user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        if db_user.password_hash != hash_password(user.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create token
        token = create_token(user.username)
        
        return TokenResponse(
            access_token=token,
            username=user.username
        )
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.post("/api/keys")
async def update_api_key(
    key_data: UpdateApiKey,
    username: str = Depends(verify_token)
):
    """Update API key for a specific model"""
    try:
        # Get current user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update API keys
        api_keys = user.api_keys or {}
        api_keys[key_data.model_name] = key_data.api_key
        
        # Update user in database
        query = users_table.update().where(users_table.c.username == username).values(api_keys=api_keys)
        await database.execute(query)
        
        return {"message": f"API key for {key_data.model_name} updated successfully"}
    except Exception as e:
        print(f"API key update error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update API key: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    username: str = Depends(verify_token)
):
    """Chat with Yappy AI"""
    try:
        print(f"Chat request from {username}: {request.message[:50]}...")
        
        # Get user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get API key for the model
        api_keys = user.api_keys or {}
        api_key = api_keys.get(request.model_name)
        
        # Get or create conversation
        conv_id = request.conversation_id or str(uuid.uuid4())
        
        # Get existing conversation
        conv_query = conversations_table.select().where(conversations_table.c.id == conv_id)
        conversation = await database.fetch_one(conv_query)
        
        if not conversation:
            # Create new conversation
            conv_data = {
                "id": conv_id,
                "user_id": user.id,
                "title": request.message[:50] + "..." if len(request.message) > 50 else request.message,
                "messages": []
            }
            insert_query = conversations_table.insert().values(**conv_data)
            await database.execute(insert_query)
            conversation_messages = []
        else:
            # Verify conversation belongs to user
            if conversation.user_id != user.id:
                raise HTTPException(status_code=403, detail="Access denied to this conversation")
            conversation_messages = conversation.messages or []
        
        # Get LLM response
        response_text, tokens = await llm_handler.get_response(
            request.message,
            request.model_name,
            api_key,
            conversation_messages
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
        
        # Update conversation
        conversation_messages.append(message_data)
        update_query = conversations_table.update().where(
            conversations_table.c.id == conv_id
        ).values(messages=conversation_messages)
        await database.execute(update_query)
        
        return ChatResponse(
            response=response_text,
            conversation_id=conv_id,
            message_id=message_id,
            timestamp=message_data["timestamp"],
            model_used=request.model_name,
            tokens_used=tokens
        )
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)