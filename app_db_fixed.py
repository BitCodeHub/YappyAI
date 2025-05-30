"""
Yappy AI Assistant - Fixed to match local app's agent routing
Web queries go to browser agent, not casual agent
"""
import os
import json
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
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
import re
import requests
import logging
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# SearxNG Search Implementation (matching local app's searxSearch.py)
class SearxSearch:
    def __init__(self):
        self.base_url = os.getenv("SEARXNG_BASE_URL", "https://search.ononoki.org")
        self.user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
        
    def execute(self, query: str) -> str:
        """Execute search matching local app's format"""
        search_url = f"{self.base_url}/search"
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': self.user_agent
        }
        data = f"q={query}&categories=general&language=auto&time_range=&safesearch=0&theme=simple".encode('utf-8')
        
        try:
            response = requests.post(search_url, headers=headers, data=data, verify=False, timeout=10)
            response.raise_for_status()
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            results = []
            
            for article in soup.find_all('article', class_='result'):
                url_header = article.find('a', class_='url_header')
                if url_header:
                    url = url_header['href']
                    title = article.find('h3').text.strip() if article.find('h3') else "No Title"
                    description = article.find('p', class_='content').text.strip() if article.find('p', class_='content') else "No Description"
                    results.append(f"Title:{title}\nSnippet:{description}\nLink:{url}")
            
            if len(results) == 0:
                # Fallback to other search methods
                return self._fallback_search(query)
                
            return "\n\n".join(results)
        except Exception as e:
            logger.error(f"SearxNG search failed: {e}")
            return self._fallback_search(query)
    
    def _fallback_search(self, query: str) -> str:
        """Fallback search methods"""
        try:
            # Try DuckDuckGo instant answers
            from urllib.parse import quote
            ddg_url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1"
            response = requests.get(ddg_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if data.get('Abstract'):
                    results.append(f"Title:Summary\nSnippet:{data['Abstract']}\nLink:{data.get('AbstractURL', '')}")
                
                if data.get('Answer'):
                    results.append(f"Title:Quick Answer\nSnippet:{data['Answer']}\nLink:")
                
                if results:
                    return "\n\n".join(results)
        except Exception as e:
            logger.error(f"Fallback search error: {e}")
        
        return "No search results found. Please try a different query."

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./yappy.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

database = Database(DATABASE_URL)
metadata = MetaData()

# Database tables
users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(50), unique=True, index=True, nullable=False),
    Column("email", String(100)),
    Column("password_hash", String(255), nullable=False),
    Column("api_keys", JSON, default={}),
    Column("preferences", JSON, default={}),
    Column("created_at", DateTime, default=datetime.now),
)

conversations_table = Table(
    "conversations",
    metadata,
    Column("id", String(50), primary_key=True),
    Column("user_id", Integer, nullable=False),
    Column("title", String(200)),
    Column("messages", JSON, default=[]),
    Column("created_at", DateTime, default=datetime.now),
    Column("updated_at", DateTime, default=datetime.now, onupdate=datetime.now)
)

# Agent Router (matching local app's logic)
class AgentRouter:
    """Routes queries to appropriate agents based on local app's router.py"""
    
    def route_query(self, query: str) -> Tuple[str, bool]:
        """
        Route query to appropriate agent
        Returns: (agent_type, needs_search)
        """
        query_lower = query.lower()
        
        # Browser agent patterns (needs web search)
        browser_patterns = [
            # Direct web requests
            'search', 'find', 'look up', 'browse', 'web', 'online', 'internet',
            # Information queries
            'who is', 'what is', 'when is', 'where is', 'how much', 'how many',
            'latest', 'current', 'recent', 'news', 'update',
            # Specific topics
            'weather', 'temperature', 'forecast',
            'price', 'cost', 'stock', 'crypto', 'bitcoin',
            'president', 'ceo', 'company', 'organization',
            'game', 'score', 'sports', 'nba', 'nfl',
            # Research
            'research', 'information about', 'tell me about', 'facts about'
        ]
        
        # Code agent patterns
        code_patterns = [
            'code', 'script', 'program', 'function', 'debug', 'error',
            'python', 'javascript', 'java', 'c++', 'sql', 'bash',
            'algorithm', 'implement', 'fix', 'syntax'
        ]
        
        # File agent patterns
        file_patterns = [
            'file', 'folder', 'directory', 'document',
            'create file', 'read file', 'find file', 'locate'
        ]
        
        # Check patterns
        if any(pattern in query_lower for pattern in browser_patterns):
            return "browser", True
        elif any(pattern in query_lower for pattern in code_patterns):
            return "code", False
        elif any(pattern in query_lower for pattern in file_patterns):
            return "file", False
        else:
            # Default to casual for general conversation
            return "casual", False

# Browser Agent Implementation
class BrowserAgent:
    """Browser agent with web search capabilities"""
    
    def __init__(self):
        self.searx_tool = SearxSearch()
        self.date = datetime.now().strftime("%B %d, %Y")
        
    async def process(self, query: str, llm_handler, api_key: str, model_name: str, conversation_history: List[Dict] = None) -> str:
        """Process query with web search"""
        
        # Perform web search
        logger.info(f"Browser agent searching for: {query}")
        search_results = self.searx_tool.execute(query)
        
        # Build context for LLM
        system_prompt = f"""You are Yappy 🐕, a friendly AI assistant with web browsing capabilities.
Today's date is {self.date}.
You have searched the web and found the following results.
Use these search results to provide accurate, current information.
Be friendly and use dog-related expressions occasionally."""
        
        user_prompt = f"""User question: {query}

Web search results:
{search_results}

Based on the search results above, please provide a helpful and accurate answer.
If the search results don't contain enough information, mention what you found and suggest the user search for more specific terms."""
        
        # Get response from LLM
        return await llm_handler._call_llm(system_prompt, user_prompt, model_name, api_key, conversation_history)

# Casual Agent Implementation
class CasualAgent:
    """Casual conversation agent without web search"""
    
    async def process(self, query: str, llm_handler, api_key: str, model_name: str, conversation_history: List[Dict] = None) -> str:
        """Process casual conversation"""
        
        system_prompt = """You are Yappy 🐕, a friendly and enthusiastic AI assistant!
Be cheerful and helpful. Use dog-related expressions occasionally (like "Woof!" or "*wags tail*").
Always aim to brighten the user's day with your positive energy!"""
        
        return await llm_handler._call_llm(system_prompt, query, model_name, api_key, conversation_history)

# LLM Handler
class LLMHandler:
    """Handles LLM interactions"""
    
    def __init__(self):
        self.router = AgentRouter()
        self.agents = {
            "browser": BrowserAgent(),
            "casual": CasualAgent()
        }
        
    async def get_response(self, prompt: str, model_name: str, api_key: Optional[str] = None, 
                          conversation_history: List[Dict] = None) -> Tuple[str, int]:
        """Get response using appropriate agent"""
        
        # Route to appropriate agent
        agent_type, needs_search = self.router.route_query(prompt)
        logger.info(f"Routing to {agent_type} agent, needs_search: {needs_search}")
        
        # Get agent
        agent = self.agents.get(agent_type, self.agents["casual"])
        
        # Process with agent
        response = await agent.process(prompt, self, api_key, model_name, conversation_history)
        
        return response, 0  # Token count would be calculated by LLM
    
    async def _call_llm(self, system_prompt: str, user_prompt: str, model_name: str, 
                        api_key: Optional[str], conversation_history: List[Dict] = None) -> str:
        """Call the appropriate LLM"""
        
        if not api_key:
            return "Woof! 🐕 I need an API key to help you! Please add one in your profile settings."
        
        try:
            if model_name == "openai" and openai:
                client = openai.OpenAI(api_key=api_key)
                
                messages = [{"role": "system", "content": system_prompt}]
                
                if conversation_history:
                    for msg in conversation_history[-5:]:
                        messages.append({"role": "user", "content": msg.get("user_message", "")})
                        messages.append({"role": "assistant", "content": msg.get("assistant_response", "")})
                
                messages.append({"role": "user", "content": user_prompt})
                
                response = client.chat.completions.create(
                    model="gpt-4" if "gpt-4" in api_key else "gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                return response.choices[0].message.content
            
            elif model_name == "anthropic" and anthropic:
                client = anthropic.Anthropic(api_key=api_key)
                
                messages = []
                if conversation_history:
                    for msg in conversation_history[-5:]:
                        messages.append({"role": "user", "content": msg.get("user_message", "")})
                        messages.append({"role": "assistant", "content": msg.get("assistant_response", "")})
                
                messages.append({"role": "user", "content": user_prompt})
                
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    system=system_prompt,
                    messages=messages,
                    max_tokens=1000
                )
                
                return response.content[0].text
            
            elif model_name == "google" and genai:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                full_prompt = system_prompt + "\n\n"
                if conversation_history:
                    for msg in conversation_history[-5:]:
                        full_prompt += f"User: {msg.get('user_message', '')}\n"
                        full_prompt += f"Assistant: {msg.get('assistant_response', '')}\n"
                
                full_prompt += f"User: {user_prompt}\nAssistant:"
                
                response = model.generate_content(full_prompt)
                return response.text
            
            elif model_name == "groq" and Groq:
                client = Groq(api_key=api_key)
                
                messages = [{"role": "system", "content": system_prompt}]
                if conversation_history:
                    for msg in conversation_history[-5:]:
                        messages.append({"role": "user", "content": msg.get("user_message", "")})
                        messages.append({"role": "assistant", "content": msg.get("assistant_response", "")})
                
                messages.append({"role": "user", "content": user_prompt})
                
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                return response.choices[0].message.content
            
            else:
                return f"Woof! 🐕 The {model_name} model isn't available. Try another one!"
                
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return f"Woof! 🐕 I encountered an error: {str(e)}"

# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🐕 Woof! Yappy AI (Fixed Agent Routing) is starting up...")
    print(f"Current date: {datetime.now().strftime('%A, %B %d, %Y')}")
    
    try:
        await database.connect()
        print("✅ Database connected successfully")
        
        engine = sqlalchemy.create_engine(DATABASE_URL)
        metadata.create_all(bind=engine)
        print("✅ Database tables created/verified")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
    
    yield
    
    # Shutdown
    print("🐕 Yappy is shutting down... Goodbye!")
    if database.is_connected:
        await database.disconnect()

app = FastAPI(title="Yappy AI Assistant", version="7.0.0", lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Security
security = HTTPBearer()

# Pydantic models
class UserLogin(BaseModel):
    username: str
    password: str

class UserSignup(BaseModel):
    username: str
    email: str
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
    agent_used: Optional[str] = None

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
        
        if not database.is_connected:
            return username
        
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        if not user:
            return username
        
        return username
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        try:
            return credentials.credentials.split(":")[0]
        except:
            raise HTTPException(status_code=401, detail="Invalid token")

# Initialize components
llm_handler = LLMHandler()

# API Endpoints
@app.get("/")
async def root():
    """Redirect to chat interface"""
    yappy_path = os.path.join(static_dir, "yappy.html")
    if os.path.exists(yappy_path):
        return FileResponse(yappy_path)
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return RedirectResponse(url="/static/yappy.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "7.0.0",
        "current_date": datetime.now().strftime("%B %d, %Y"),
        "features": ["browser_agent", "casual_agent", "web_search"],
        "database": "connected" if database.is_connected else "disconnected"
    }

@app.post("/auth/register", response_model=TokenResponse)
async def register(user_data: UserSignup):
    """Register new user"""
    try:
        query = users_table.select().where(users_table.c.username == user_data.username)
        existing_user = await database.fetch_one(query)
        
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        query = users_table.insert().values(
            username=user_data.username,
            email=user_data.email,
            password_hash=hash_password(user_data.password),
            created_at=datetime.now(),
            api_keys={},
            preferences={}
        )
        
        await database.execute(query)
        
        token = create_token(user_data.username)
        
        return TokenResponse(
            access_token=token,
            username=user_data.username
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/auth/login", response_model=TokenResponse)
async def login(user_data: UserLogin):
    """Login user"""
    try:
        query = users_table.select().where(users_table.c.username == user_data.username)
        user = await database.fetch_one(query)
        
        if not user or user.password_hash != hash_password(user_data.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        token = create_token(user_data.username)
        
        return TokenResponse(
            access_token=token,
            username=user_data.username
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, username: str = Depends(verify_token)):
    """Main chat endpoint with proper agent routing"""
    try:
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
        
        # Get response from appropriate agent
        response_text, tokens = await llm_handler.get_response(
            request.message,
            request.model_name,
            api_key,
            conversation_messages
        )
        
        # Determine which agent was used
        agent_type, _ = llm_handler.router.route_query(request.message)
        
        # Store message
        message_id = str(uuid.uuid4())
        message_data = {
            "id": message_id,
            "user_message": request.message,
            "assistant_response": response_text,
            "model": request.model_name,
            "timestamp": datetime.now().isoformat(),
            "tokens": tokens,
            "agent_used": agent_type
        }
        
        # Update conversation
        conversation_messages.append(message_data)
        
        # Handle missing updated_at column
        try:
            update_query = conversations_table.update().where(
                conversations_table.c.id == conv_id
            ).values(
                messages=conversation_messages,
                updated_at=datetime.now()
            )
            await database.execute(update_query)
        except Exception as e:
            logger.warning(f"Updated_at column issue: {e}")
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
            tokens_used=tokens,
            agent_used=agent_type
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/api/user/api-key")
async def update_api_key(update_data: UpdateApiKey, username: str = Depends(verify_token)):
    """Update user's API key for a specific model"""
    try:
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        api_keys = user.api_keys or {}
        api_keys[update_data.model_name] = update_data.api_key
        
        update_query = users_table.update().where(
            users_table.c.username == username
        ).values(api_keys=api_keys)
        
        await database.execute(update_query)
        
        return {"status": "success", "message": f"API key updated for {update_data.model_name}"}
        
    except Exception as e:
        logger.error(f"API key update error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update API key")

@app.get("/api/conversations")
async def get_conversations(username: str = Depends(verify_token)):
    """Get user's conversations"""
    try:
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            return []
        
        conv_query = conversations_table.select().where(
            conversations_table.c.user_id == user.id
        ).order_by(conversations_table.c.created_at.desc())
        
        conversations = await database.fetch_all(conv_query)
        
        result = []
        for conv in conversations:
            messages = conv.messages or []
            last_message = messages[-1] if messages else None
            
            result.append({
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
                "message_count": len(messages),
                "last_message": last_message.get("user_message") if last_message else None,
                "agent_used": last_message.get("agent_used", "casual") if last_message else None
            })
        
        return result
    except Exception as e:
        logger.error(f"Get conversations error: {e}")
        return []

@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str, username: str = Depends(verify_token)):
    """Get specific conversation"""
    try:
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        conv_query = conversations_table.select().where(
            conversations_table.c.id == conversation_id
        )
        conversation = await database.fetch_one(conv_query)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        if conversation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            "id": conversation.id,
            "title": conversation.title,
            "messages": conversation.messages or [],
            "created_at": conversation.created_at.isoformat() if conversation.created_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get conversation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get conversation")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)