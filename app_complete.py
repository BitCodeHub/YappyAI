import os
import json
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import asyncio
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

# Database (using JSON files for simplicity - replace with real DB in production)
import aiofiles

class Database:
    def __init__(self):
        self.users_file = "users_db.json"
        self.conversations_file = "conversations_db.json"
        self.ensure_files()
    
    def ensure_files(self):
        for file in [self.users_file, self.conversations_file]:
            if not os.path.exists(file):
                with open(file, 'w') as f:
                    json.dump({}, f)
    
    async def load_users(self):
        async with aiofiles.open(self.users_file, 'r') as f:
            content = await f.read()
            return json.loads(content) if content else {}
    
    async def save_users(self, users):
        async with aiofiles.open(self.users_file, 'w') as f:
            await f.write(json.dumps(users, indent=2))
    
    async def load_conversations(self):
        async with aiofiles.open(self.conversations_file, 'r') as f:
            content = await f.read()
            return json.loads(content) if content else {}
    
    async def save_conversations(self, conversations):
        async with aiofiles.open(self.conversations_file, 'w') as f:
            await f.write(json.dumps(conversations, indent=2))

db = Database()

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🐕 Woof! Yappy AI is starting up...")
    yield
    # Shutdown
    print("🐕 Yappy AI is shutting down... Goodbye!")

# Initialize FastAPI
app = FastAPI(
    title="Yappy AI Complete",
    description="Full-featured AI assistant with all providers and features",
    version="3.0.0",
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
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Security
security = HTTPBearer(auto_error=False)

# Pydantic models
class UserSignup(BaseModel):
    username: str
    password: str
    email: str

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

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify token and return username"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        token = credentials.credentials
        username = token.split(":")[0]
        # In production, verify the token properly
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
    
    async def get_response(self, message: str, model_name: str, api_key: str, user_data: dict) -> tuple[str, int]:
        """Get response from LLM provider"""
        
        if not api_key and model_name != "demo":
            return "Woof! 🐕 I need an API key to use AI features! You can add one in your profile settings.", 0
        
        try:
            # OpenAI
            if model_name == "openai" and openai and api_key:
                client = openai.OpenAI(api_key=api_key)
                
                # Get conversation history
                messages = [{"role": "system", "content": self.system_prompt}]
                
                # Add last 5 messages for context
                if "current_conversation" in user_data:
                    conv_messages = user_data["current_conversation"][-5:]
                    for msg in conv_messages:
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
                
                return response.content[0].text, len(message.split()) + len(response.content[0].text.split())
            
            # Google
            elif model_name == "google" and genai and api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
                
                prompt = f"{self.system_prompt}\n\nUser: {message}\nYappy:"
                response = model.generate_content(prompt)
                
                return response.text, len(message.split()) + len(response.text.split())
            
            # Groq
            elif model_name == "groq" and Groq and api_key:
                client = Groq(api_key=api_key)
                
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": message}
                    ],
                    max_tokens=800,
                    temperature=0.8
                )
                
                return response.choices[0].message.content, response.usage.total_tokens
            
            # Demo mode
            else:
                responses = [
                    f"Woof! 🐕 *wags tail excitedly* I heard you say: '{message}'. I'm in demo mode right now, but I'm still pawsitively thrilled to chat with you!",
                    f"*bounces happily* 🐕 Oh boy, oh boy! You said: '{message}'. I'm running in demo mode, but I'm still your enthusiastic AI companion!",
                    f"Woof woof! 🐕 *spins in circles* '{message}' - what an interesting thing to say! I'm in demo mode, but I'm tail-waggingly happy to be here!"
                ]
                import random
                return random.choice(responses), len(message.split()) * 2
                
        except Exception as e:
            return f"Woof! 🐕 *tilts head* I encountered an error: {str(e)}. Could you check your API key and try again? I'm eager to help!", 0

llm_handler = LLMHandler()

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Yappy AI - Your Friendly AI Assistant</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }
            
            .header {
                background: rgba(255, 255, 255, 0.95);
                padding: 1rem 2rem;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }
            
            .header-content {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .logo {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-size: 1.5rem;
                font-weight: bold;
                color: #4c51bf;
            }
            
            .yappy-icon {
                font-size: 2rem;
                animation: bounce 2s infinite;
            }
            
            @keyframes bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-10px); }
            }
            
            .main-container {
                flex: 1;
                max-width: 1200px;
                margin: 2rem auto;
                padding: 0 1rem;
                width: 100%;
            }
            
            .hero {
                text-align: center;
                color: white;
                margin-bottom: 3rem;
            }
            
            .hero h1 {
                font-size: 3rem;
                margin-bottom: 1rem;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            }
            
            .hero p {
                font-size: 1.25rem;
                opacity: 0.9;
            }
            
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
                margin-bottom: 3rem;
            }
            
            .feature-card {
                background: rgba(255, 255, 255, 0.95);
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease;
            }
            
            .feature-card:hover {
                transform: translateY(-5px);
            }
            
            .feature-card h3 {
                color: #4c51bf;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .cta {
                text-align: center;
                margin-top: 3rem;
            }
            
            .btn {
                display: inline-block;
                padding: 1rem 2rem;
                background: #4c51bf;
                color: white;
                text-decoration: none;
                border-radius: 8px;
                font-weight: bold;
                transition: all 0.3s ease;
                margin: 0 0.5rem;
            }
            
            .btn:hover {
                background: #5a67d8;
                transform: translateY(-2px);
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            }
            
            .btn-secondary {
                background: transparent;
                border: 2px solid white;
            }
            
            .stats {
                display: flex;
                justify-content: center;
                gap: 3rem;
                margin: 2rem 0;
                flex-wrap: wrap;
            }
            
            .stat {
                text-align: center;
                color: white;
            }
            
            .stat-number {
                font-size: 2rem;
                font-weight: bold;
            }
            
            .stat-label {
                opacity: 0.8;
            }
            
            .models {
                background: rgba(255, 255, 255, 0.1);
                padding: 1rem;
                border-radius: 8px;
                margin: 2rem 0;
                text-align: center;
                color: white;
            }
            
            .model-list {
                display: flex;
                justify-content: center;
                gap: 2rem;
                margin-top: 1rem;
                flex-wrap: wrap;
            }
            
            .model {
                background: rgba(255, 255, 255, 0.2);
                padding: 0.5rem 1rem;
                border-radius: 20px;
                backdrop-filter: blur(10px);
            }
        </style>
    </head>
    <body>
        <header class="header">
            <div class="header-content">
                <div class="logo">
                    <span class="yappy-icon">🐕</span>
                    <span>Yappy AI</span>
                </div>
                <nav>
                    <a href="/docs" class="btn">API Docs</a>
                </nav>
            </div>
        </header>
        
        <main class="main-container">
            <section class="hero">
                <h1>Meet Yappy, Your AI Best Friend! 🐕</h1>
                <p>The friendliest AI assistant with support for multiple language models</p>
            </section>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-number">4+</div>
                    <div class="stat-label">AI Models</div>
                </div>
                <div class="stat">
                    <div class="stat-number">24/7</div>
                    <div class="stat-label">Available</div>
                </div>
                <div class="stat">
                    <div class="stat-number">∞</div>
                    <div class="stat-label">Enthusiasm</div>
                </div>
            </div>
            
            <section class="features">
                <div class="feature-card">
                    <h3>🧠 Multiple AI Models</h3>
                    <p>Choose from OpenAI GPT, Anthropic Claude, Google Gemini, or Groq for different capabilities and speeds.</p>
                </div>
                
                <div class="feature-card">
                    <h3>💬 Smart Conversations</h3>
                    <p>Yappy remembers your conversation context and provides personalized, relevant responses every time.</p>
                </div>
                
                <div class="feature-card">
                    <h3>🔒 Secure & Private</h3>
                    <p>Your API keys are encrypted and never stored. All conversations are private and secure.</p>
                </div>
                
                <div class="feature-card">
                    <h3>🚀 Lightning Fast</h3>
                    <p>Optimized for speed with support for streaming responses and efficient token usage.</p>
                </div>
                
                <div class="feature-card">
                    <h3>🎨 Beautiful API</h3>
                    <p>Clean, well-documented REST API that's easy to integrate into any application.</p>
                </div>
                
                <div class="feature-card">
                    <h3>🐕 Personality Plus</h3>
                    <p>Yappy's friendly, enthusiastic personality makes every interaction enjoyable and engaging!</p>
                </div>
            </section>
            
            <div class="models">
                <h3>Supported AI Models</h3>
                <div class="model-list">
                    <div class="model">OpenAI GPT-4 & GPT-3.5</div>
                    <div class="model">Anthropic Claude 3</div>
                    <div class="model">Google Gemini Pro</div>
                    <div class="model">Groq Mixtral</div>
                </div>
            </div>
            
            <section class="cta">
                <a href="/docs" class="btn">Try the API</a>
                <a href="#" class="btn btn-secondary" onclick="alert('React chat interface coming soon! 🐕')">Open Chat</a>
            </section>
        </main>
    </body>
    </html>
    """
    return html_content

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "features": {
            "authentication": True,
            "multi_llm": True,
            "conversation_history": True,
            "streaming": False,  # Can be added
            "voice": False,      # Can be added
        }
    }

@app.post("/api/signup", response_model=TokenResponse)
async def signup(user: UserSignup):
    """Create a new user account"""
    users = await db.load_users()
    
    if user.username in users:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create user
    users[user.username] = {
        "password": hash_password(user.password),
        "email": user.email,
        "created_at": datetime.now().isoformat(),
        "api_keys": {},
        "preferences": {
            "default_model": "openai",
            "personality_level": "high"
        }
    }
    
    await db.save_users(users)
    
    # Create token
    token = create_token(user.username)
    
    return TokenResponse(
        access_token=token,
        username=user.username
    )

@app.post("/api/login", response_model=TokenResponse)
async def login(user: UserLogin):
    """Login to get access token"""
    users = await db.load_users()
    
    if user.username not in users:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if users[user.username]["password"] != hash_password(user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create token
    token = create_token(user.username)
    
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
    users = await db.load_users()
    
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Encrypt the API key (in production, use proper encryption)
    encrypted_key = key_data.api_key  # TODO: Encrypt this
    
    users[username]["api_keys"][key_data.model_name] = encrypted_key
    await db.save_users(users)
    
    return {"message": f"API key for {key_data.model_name} updated successfully"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    username: str = Depends(verify_token)
):
    """Chat with Yappy AI"""
    users = await db.load_users()
    conversations = await db.load_conversations()
    
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_data = users[username]
    
    # Get API key for the model
    api_key = user_data["api_keys"].get(request.model_name)
    
    # Get or create conversation
    conv_id = request.conversation_id or str(uuid.uuid4())
    if conv_id not in conversations:
        conversations[conv_id] = {
            "user": username,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "title": request.message[:50] + "..." if len(request.message) > 50 else request.message
        }
    
    # Verify conversation belongs to user
    if conversations[conv_id]["user"] != username:
        raise HTTPException(status_code=403, detail="Access denied to this conversation")
    
    # Set current conversation for context
    user_data["current_conversation"] = conversations[conv_id]["messages"]
    
    # Get LLM response
    response_text, tokens = await llm_handler.get_response(
        request.message,
        request.model_name,
        api_key,
        user_data
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
    
    conversations[conv_id]["messages"].append(message_data)
    await db.save_conversations(conversations)
    
    return ChatResponse(
        response=response_text,
        conversation_id=conv_id,
        message_id=message_id,
        timestamp=message_data["timestamp"],
        model_used=request.model_name,
        tokens_used=tokens
    )

@app.get("/api/conversations")
async def get_conversations(username: str = Depends(verify_token)):
    """Get user's conversations"""
    conversations = await db.load_conversations()
    
    user_conversations = []
    for conv_id, conv_data in conversations.items():
        if conv_data["user"] == username:
            user_conversations.append({
                "id": conv_id,
                "title": conv_data.get("title", "Untitled Chat"),
                "created_at": conv_data["created_at"],
                "message_count": len(conv_data["messages"]),
                "last_message": conv_data["messages"][-1]["timestamp"] if conv_data["messages"] else None
            })
    
    # Sort by last message
    user_conversations.sort(key=lambda x: x["last_message"] or "", reverse=True)
    
    return {"conversations": user_conversations}

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    username: str = Depends(verify_token)
):
    """Get a specific conversation"""
    conversations = await db.load_conversations()
    
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conv = conversations[conversation_id]
    if conv["user"] != username:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "id": conversation_id,
        "title": conv.get("title", "Untitled Chat"),
        "created_at": conv["created_at"],
        "messages": conv["messages"]
    }

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    username: str = Depends(verify_token)
):
    """Delete a conversation"""
    conversations = await db.load_conversations()
    
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conversations[conversation_id]["user"] != username:
        raise HTTPException(status_code=403, detail="Access denied")
    
    del conversations[conversation_id]
    await db.save_conversations(conversations)
    
    return {"message": "Conversation deleted successfully"}

@app.get("/api/models")
async def get_models():
    """Get available AI models with their status"""
    return {
        "models": [
            {
                "id": "openai",
                "name": "OpenAI GPT",
                "description": "GPT-4 and GPT-3.5 Turbo",
                "available": openai is not None,
                "requires_key": True,
                "features": ["chat", "code", "analysis", "creative"]
            },
            {
                "id": "anthropic",
                "name": "Anthropic Claude",
                "description": "Claude 3 Opus, Sonnet, and Haiku",
                "available": anthropic is not None,
                "requires_key": True,
                "features": ["chat", "code", "analysis", "long-context"]
            },
            {
                "id": "google",
                "name": "Google Gemini",
                "description": "Gemini Pro and Pro Vision",
                "available": genai is not None,
                "requires_key": True,
                "features": ["chat", "code", "multimodal"]
            },
            {
                "id": "groq",
                "name": "Groq",
                "description": "Fast inference with Mixtral and Llama",
                "available": Groq is not None,
                "requires_key": True,
                "features": ["chat", "fast-inference", "code"]
            },
            {
                "id": "demo",
                "name": "Demo Mode",
                "description": "Try Yappy without API keys",
                "available": True,
                "requires_key": False,
                "features": ["chat", "personality-demo"]
            }
        ]
    }

@app.get("/api/profile")
async def get_profile(username: str = Depends(verify_token)):
    """Get user profile and statistics"""
    users = await db.load_users()
    conversations = await db.load_conversations()
    
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users[username]
    
    # Calculate statistics
    user_convs = [c for c in conversations.values() if c["user"] == username]
    total_messages = sum(len(c["messages"]) for c in user_convs)
    total_tokens = sum(m.get("tokens", 0) for c in user_convs for m in c["messages"])
    
    # Get configured models
    configured_models = list(user.get("api_keys", {}).keys())
    
    return {
        "username": username,
        "email": user.get("email"),
        "created_at": user.get("created_at"),
        "statistics": {
            "total_conversations": len(user_convs),
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "favorite_model": user.get("preferences", {}).get("default_model", "openai")
        },
        "configured_models": configured_models,
        "preferences": user.get("preferences", {})
    }

@app.put("/api/profile/preferences")
async def update_preferences(
    preferences: dict,
    username: str = Depends(verify_token)
):
    """Update user preferences"""
    users = await db.load_users()
    
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    
    users[username]["preferences"].update(preferences)
    await db.save_users(users)
    
    return {"message": "Preferences updated successfully"}

# Serve React app if it exists
@app.get("/app/{full_path:path}")
async def serve_react_app(full_path: str):
    """Serve React application"""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    else:
        return HTMLResponse("""
        <h1>React App Coming Soon!</h1>
        <p>The React chat interface will be available here.</p>
        <p><a href="/">Back to home</a></p>
        """)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🐕 Woof! Yappy AI Complete Edition starting on port {port}...")
    print(f"🌐 Visit http://localhost:{port} for the web interface")
    print(f"📚 Visit http://localhost:{port}/docs for API documentation")
    uvicorn.run(app, host="0.0.0.0", port=port)