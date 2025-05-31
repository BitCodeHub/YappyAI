"""
AgenticSeek/Yappy Web Version with Agent Routing System
Implements the exact agent architecture from the local app
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
import logging

# Import the SearxNG search tool
from searx_search_tool import SearxSearch, fallback_web_search

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

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./yappy.db")

# Fix for Render PostgreSQL URLs
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create database instance
database = Database(DATABASE_URL)
metadata = MetaData()

# Define tables
users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String(50), unique=True, nullable=False),
    Column("email", String(100), unique=True, nullable=False),
    Column("password_hash", String(256), nullable=False),
    Column("api_keys", JSON, default={}),
    Column("preferences", JSON, default={}),
    Column("created_at", DateTime, default=datetime.now)
)

conversations_table = Table(
    "conversations",
    metadata,
    Column("id", String(50), primary_key=True),
    Column("user_id", Integer),
    Column("title", String(200)),
    Column("messages", JSON, default=[]),
    Column("created_at", DateTime, default=datetime.now),
    Column("updated_at", DateTime, default=datetime.now, onupdate=datetime.now)
)

# Agent Router - Mimics the local app's router.py logic
class AgentRouter:
    """Routes queries to appropriate agents based on content analysis"""
    
    def __init__(self):
        self.searx_tool = SearxSearch()
    
    def analyze_query(self, query: str, context: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze query and determine which agent to use
        Returns: (agent_type, analysis_data)
        """
        query_lower = query.lower()
        
        # Web/Browser agent patterns - needs real-time data
        web_patterns = [
            # Current information
            'latest', 'current', 'today', 'now', 'recent', 'update', 
            'news', 'happening', 'trend', 'breaking',
            # Weather
            'weather', 'temperature', 'forecast', 'rain', 'snow', 'climate',
            # Sports/Events
            'game', 'score', 'match', 'playing', 'nba', 'nfl', 'soccer', 'sports',
            # Financial
            'price', 'stock', 'crypto', 'bitcoin', 'btc', 'market', 'trading',
            # People/Organizations
            'president', 'ceo', 'founder', 'minister', 'leader', 'company',
            # Search indicators
            'search', 'find', 'look up', 'google', 'check',
            # Questions needing current data
            'who is', 'what is the', 'when is', 'where is', 'how much',
            'is there', 'are there'
        ]
        
        # Code agent patterns
        code_patterns = [
            'code', 'program', 'function', 'debug', 'error', 'implement',
            'python', 'javascript', 'java', 'c++', 'sql', 'api',
            'algorithm', 'compile', 'syntax', 'variable', 'class'
        ]
        
        # File agent patterns
        file_patterns = [
            'file', 'folder', 'directory', 'create file', 'read file',
            'write file', 'delete', 'rename', 'move', 'copy'
        ]
        
        # Check patterns
        if any(pattern in query_lower for pattern in web_patterns):
            return "browser", {"needs_search": True, "query": query}
        elif any(pattern in query_lower for pattern in code_patterns):
            return "code", {"query": query}
        elif any(pattern in query_lower for pattern in file_patterns):
            return "file", {"query": query}
        else:
            # Default to casual agent for general conversation
            return "casual", {"query": query}

# Agent implementations
class BaseAgent:
    """Base agent class"""
    
    def __init__(self, name: str, llm_handler):
        self.name = name
        self.llm_handler = llm_handler
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Get agent-specific system prompt"""
        return f"You are {self.name} agent. You help users with {self.name}-related tasks."
    
    async def process(self, query: str, api_key: str, model_name: str, 
                      conversation_history: List[Dict] = None, **kwargs) -> str:
        """Process query and return response"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                messages.append({"role": "user", "content": msg.get("user_message", "")})
                messages.append({"role": "assistant", "content": msg.get("assistant_response", "")})
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        # Get response from LLM
        response, tokens = await self.llm_handler.get_response_with_messages(
            messages, model_name, api_key
        )
        
        return response

class BrowserAgent(BaseAgent):
    """Browser agent with web search capabilities"""
    
    def __init__(self, llm_handler):
        super().__init__("Browser", llm_handler)
        self.searx_tool = SearxSearch()
        self.system_prompt = """You are a web browser agent with real-time internet access. 
Your role is to search the web for current information and provide accurate, up-to-date responses.
Always use the search results provided to give factual, current information.
If you receive search results, use them to answer the user's question.
Never make up information - only use what's in the search results or clearly state if you don't have the information."""
    
    async def process(self, query: str, api_key: str, model_name: str, 
                      conversation_history: List[Dict] = None, **kwargs) -> str:
        """Process query with web search"""
        
        # Perform web search
        search_results = self.searx_tool.search(query)
        
        # Fallback to DuckDuckGo if no results
        if not search_results:
            search_results = fallback_web_search(query)
        
        # Format search results
        web_context = ""
        if search_results:
            web_context = "\n\nWeb Search Results:\n"
            web_context += self.searx_tool.format_results(search_results)
            logger.info(f"Found {len(search_results)} search results for: {query}")
        else:
            logger.warning(f"No search results found for: {query}")
        
        # Prepare enhanced query with search results
        enhanced_query = f"""{query}

{web_context}

Based on the search results above, please provide an accurate answer to the user's question.
Today's date is {datetime.now().strftime('%B %d, %Y')}.
Use the search results to give current, factual information."""
        
        # Get response from parent class
        return await super().process(
            enhanced_query, api_key, model_name, conversation_history, **kwargs
        )

class CasualAgent(BaseAgent):
    """Casual conversation agent"""
    
    def __init__(self, llm_handler):
        super().__init__("Casual", llm_handler)
        self.system_prompt = """You are Yappy 🐕, a friendly and enthusiastic AI assistant!
You love to help users with general questions and casual conversation.
Be cheerful, use dog-related expressions occasionally (like "Woof!" or "*wags tail*"), 
and always aim to brighten the user's day.
You're knowledgeable but approachable. Keep responses conversational and friendly."""

class CodeAgent(BaseAgent):
    """Code assistance agent"""
    
    def __init__(self, llm_handler):
        super().__init__("Code", llm_handler)
        self.system_prompt = """You are a code assistant agent specializing in programming and software development.
You help with:
- Writing and debugging code
- Explaining programming concepts
- Reviewing code for best practices
- Suggesting optimizations
- Helping with various programming languages
Always provide clear, well-commented code examples when relevant."""

# LLM Handler
class LLMHandler:
    """Handles all LLM interactions"""
    
    def __init__(self):
        self.clients = {}
        self._setup_clients()
    
    def _setup_clients(self):
        """Setup available LLM clients"""
        if openai:
            self.clients['openai'] = 'available'
        if anthropic:
            self.clients['anthropic'] = 'available'
        if genai:
            self.clients['google'] = 'available'
        if Groq:
            self.clients['groq'] = 'available'
    
    async def get_response_with_messages(self, messages: List[Dict], model_name: str, 
                                          api_key: Optional[str] = None) -> Tuple[str, int]:
        """Get response from LLM with message history"""
        
        if not api_key:
            return "Woof! 🐕 Please add an API key in your profile to unlock my full potential!", 0
        
        try:
            if model_name == "openai" and openai:
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content, response.usage.total_tokens
            
            elif model_name == "anthropic" and anthropic:
                client = anthropic.Anthropic(api_key=api_key)
                # Convert messages to Anthropic format
                system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
                other_messages = [m for m in messages if m["role"] != "system"]
                
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    system=system_msg,
                    messages=other_messages,
                    max_tokens=1000
                )
                return response.content[0].text, response.usage.input_tokens + response.usage.output_tokens
            
            elif model_name == "google" and genai:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
                
                # Convert messages to Gemini format
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                response = model.generate_content(prompt)
                return response.text, 100  # Gemini doesn't provide token count
            
            elif model_name == "groq" and Groq:
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content, response.usage.total_tokens
            
            else:
                return f"Model {model_name} not available or not configured.", 0
                
        except Exception as e:
            logger.error(f"LLM Error for {model_name}: {e}")
            return f"Error: {str(e)}", 0
    
    async def get_response(self, prompt: str, model_name: str, api_key: Optional[str] = None, 
                           conversation_history: List[Dict] = None) -> Tuple[str, int]:
        """Legacy method for compatibility"""
        messages = []
        if conversation_history:
            for msg in conversation_history[-5:]:
                messages.append({"role": "user", "content": msg.get("user_message", "")})
                messages.append({"role": "assistant", "content": msg.get("assistant_response", "")})
        messages.append({"role": "user", "content": prompt})
        
        return await self.get_response_with_messages(messages, model_name, api_key)

# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    await database.connect()
    yield
    await database.disconnect()

app = FastAPI(title="Yappy AI Assistant", version="5.0.0", lifespan=lifespan)

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

# Initialize components
llm_handler = LLMHandler()
agent_router = AgentRouter()

# Initialize agents
agents = {
    "browser": BrowserAgent(llm_handler),
    "casual": CasualAgent(llm_handler),
    "code": CodeAgent(llm_handler)
}

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
        return username
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

# API Endpoints
@app.get("/")
async def root():
    """Redirect to chat interface"""
    yappy_path = os.path.join(static_dir, "yappy.html")
    if os.path.exists(yappy_path):
        return FileResponse(yappy_path)
    return {"message": "Yappy AI Assistant API"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "5.0.0",
        "current_date": datetime.now().strftime("%B %d, %Y"),
        "features": ["multi_agent", "web_search", "context_awareness"],
        "agents": list(agents.keys()),
        "database": "connected" if database.is_connected else "disconnected"
    }

@app.post("/auth/register", response_model=TokenResponse)
async def register(user_data: UserSignup):
    """Register new user"""
    try:
        # Check if user exists
        query = users_table.select().where(users_table.c.username == user_data.username)
        existing_user = await database.fetch_one(query)
        
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Create new user
        query = users_table.insert().values(
            username=user_data.username,
            email=user_data.email,
            password_hash=hash_password(user_data.password),
            created_at=datetime.now(),
            api_keys={},
            preferences={}
        )
        
        await database.execute(query)
        
        # Create token
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
        # Find user
        query = users_table.select().where(users_table.c.username == user_data.username)
        user = await database.fetch_one(query)
        
        if not user or user.password_hash != hash_password(user_data.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create token
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
    """Main chat endpoint with agent routing"""
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
        
        # Route to appropriate agent
        agent_type, analysis_data = agent_router.analyze_query(request.message)
        logger.info(f"Routing to {agent_type} agent for query: {request.message}")
        
        # Get agent
        agent = agents.get(agent_type, agents["casual"])
        
        # Process with agent
        response_text = await agent.process(
            request.message,
            api_key,
            request.model_name,
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
            "agent_used": agent_type,
            "had_web_search": agent_type == "browser"
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
            tokens_used=0,  # TODO: Get actual token count
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
        # Get current user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update API keys
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
        # Get user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            return []
        
        # Get conversations
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
                "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
                "message_count": len(messages),
                "last_message": last_message.get("user_message") if last_message else None,
                "last_agent": last_message.get("agent_used", "casual") if last_message else None
            })
        
        return result
    except Exception as e:
        logger.error(f"Get conversations error: {e}")
        return []

@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str, username: str = Depends(verify_token)):
    """Get specific conversation"""
    try:
        # Get user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get conversation
        conv_query = conversations_table.select().where(
            conversations_table.c.id == conversation_id
        )
        conversation = await database.fetch_one(conv_query)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Verify ownership
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

# Create tables on startup
@app.on_event("startup")
async def startup():
    """Create database tables"""
    engine = sqlalchemy.create_engine(DATABASE_URL)
    metadata.create_all(engine)
    logger.info("Database tables created")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)