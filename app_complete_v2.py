import os
import json
import uuid
import hashlib
import secrets
import tempfile
import subprocess
import asyncio
import aiofiles
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from databases import Database
import sqlalchemy
from sqlalchemy import MetaData, Table, Column, Integer, String, DateTime, Text, JSON, Boolean, Float, ForeignKey
import httpx
import PyPDF2
from io import BytesIO
import re
import tiktoken

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

# Agent Types Enum
class AgentType(str, Enum):
    CASUAL = "casual"
    CODER = "coder"
    FILE = "file"
    BROWSER = "browser"
    PLANNER = "planner"
    MCP = "mcp"

# Tool Types Enum
class ToolType(str, Enum):
    PYTHON = "python"
    BASH = "bash"
    JAVA = "java"
    GO = "go"
    C = "c"
    FILE_FINDER = "file_finder"
    FLIGHT_SEARCH = "flight_search"
    WEB_SEARCH = "web_search"
    MCP_FINDER = "mcp_finder"

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./yappy_complete.db")

# Fix for Render PostgreSQL URLs
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

database = Database(DATABASE_URL)
metadata = MetaData()

# Enhanced Database Tables
users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(50), unique=True, index=True, nullable=False),
    Column("email", String(100)),
    Column("password_hash", String(255), nullable=False),
    Column("api_keys", JSON, default={}),
    Column("preferences", JSON, default={}),
    Column("feature_flags", JSON, default={}),
    Column("voice_settings", JSON, default={"tts_enabled": False, "stt_enabled": False, "voice": "default"}),
    Column("created_at", DateTime, default=datetime.now),
    Column("last_login", DateTime),
)

conversations_table = Table(
    "conversations",
    metadata,
    Column("id", String(50), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id"), nullable=False),
    Column("title", String(200)),
    Column("messages", JSON, default=[]),
    Column("agent_type", String(20), default="casual"),
    Column("metadata", JSON, default={}),
    Column("created_at", DateTime, default=datetime.now),
    Column("updated_at", DateTime, default=datetime.now),
)

agent_executions_table = Table(
    "agent_executions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("conversation_id", String(50), ForeignKey("conversations.id")),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("agent_type", String(20)),
    Column("query", Text),
    Column("response", Text),
    Column("tools_used", JSON, default=[]),
    Column("execution_time", Float),
    Column("status", String(20), default="completed"),
    Column("error_message", Text),
    Column("created_at", DateTime, default=datetime.now),
)

tool_results_table = Table(
    "tool_results",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("execution_id", Integer, ForeignKey("agent_executions.id")),
    Column("tool_type", String(30)),
    Column("input_data", JSON),
    Column("output_data", JSON),
    Column("execution_time", Float),
    Column("status", String(20)),
    Column("created_at", DateTime, default=datetime.now),
)

file_uploads_table = Table(
    "file_uploads",
    metadata,
    Column("id", String(50), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("filename", String(255)),
    Column("file_type", String(50)),
    Column("file_size", Integer),
    Column("file_path", String(500)),
    Column("metadata", JSON, default={}),
    Column("created_at", DateTime, default=datetime.now),
)

task_plans_table = Table(
    "task_plans",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("conversation_id", String(50), ForeignKey("conversations.id")),
    Column("plan_description", Text),
    Column("steps", JSON, default=[]),
    Column("current_step", Integer, default=0),
    Column("status", String(20), default="pending"),
    Column("created_at", DateTime, default=datetime.now),
    Column("completed_at", DateTime),
)

memory_compressions_table = Table(
    "memory_compressions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("conversation_id", String(50), ForeignKey("conversations.id")),
    Column("original_text", Text),
    Column("compressed_text", Text),
    Column("compression_ratio", Float),
    Column("created_at", DateTime, default=datetime.now),
)

engine = sqlalchemy.create_engine(DATABASE_URL)

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🐕 Woof! Yappy AI Complete is starting up...")
    print(f"Database URL: {DATABASE_URL[:50]}...")
    
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
    title="Yappy AI Complete - All Features",
    description="Full-featured AI assistant with all agent types and tools",
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

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
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
    agent_type: Optional[AgentType] = AgentType.CASUAL
    stream: Optional[bool] = False
    file_data: Optional[Dict[str, Any]] = None
    tools_enabled: Optional[List[str]] = None
    language: Optional[str] = "en"

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    timestamp: str
    model_used: str
    agent_used: AgentType
    tools_used: List[str] = []
    tokens_used: Optional[int] = None
    execution_time: Optional[float] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    expires_in: int = 3600

class TaskPlanRequest(BaseModel):
    task_description: str
    conversation_id: Optional[str] = None

class TaskPlanResponse(BaseModel):
    plan_id: int
    steps: List[Dict[str, Any]]
    estimated_time: Optional[float] = None

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
        
        if database.is_connected:
            query = users_table.select().where(users_table.c.username == username)
            user = await database.fetch_one(query)
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
        
        return username
    except Exception as e:
        print(f"Token verification error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

# Agent Router
class AgentRouter:
    """Intelligent agent selection based on query analysis"""
    
    def __init__(self):
        self.routing_rules = {
            "code": ["python", "javascript", "java", "code", "function", "class", "debug", "error", "implement"],
            "file": ["file", "directory", "folder", "find", "search", "locate", "read", "write"],
            "browser": ["website", "browse", "search web", "google", "url", "click", "navigate"],
            "planner": ["plan", "project", "steps", "complex", "multiple tasks", "organize", "workflow"],
            "mcp": ["mcp", "tool", "server", "protocol"],
        }
    
    async def route_query(self, query: str, file_data: Optional[Dict] = None) -> AgentType:
        """Determine the best agent for a query"""
        query_lower = query.lower()
        
        # File upload -> File agent
        if file_data:
            return AgentType.FILE
        
        # Check for explicit agent mentions
        if "code" in query_lower or "program" in query_lower:
            return AgentType.CODER
        
        # Check routing rules
        scores = {}
        for agent, keywords in self.routing_rules.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[agent] = score
        
        if scores:
            best_agent = max(scores, key=scores.get)
            return AgentType(best_agent)
        
        # Complex queries go to planner
        if len(query.split()) > 20 or "and" in query_lower and "then" in query_lower:
            return AgentType.PLANNER
        
        return AgentType.CASUAL

# Code Interpreters
class CodeInterpreter:
    """Base class for code interpreters"""
    
    def __init__(self, language: str):
        self.language = language
        self.temp_dir = tempfile.gettempdir()
    
    async def execute(self, code: str) -> Dict[str, Any]:
        """Execute code and return results"""
        raise NotImplementedError

class PythonInterpreter(CodeInterpreter):
    """Python code interpreter"""
    
    def __init__(self):
        super().__init__("python")
    
    async def execute(self, code: str) -> Dict[str, Any]:
        """Execute Python code"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute code
            process = await asyncio.create_subprocess_exec(
                'python3', temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # Clean up
            os.unlink(temp_file)
            
            return {
                "status": "success" if process.returncode == 0 else "error",
                "output": stdout.decode() if stdout else "",
                "error": stderr.decode() if stderr else "",
                "language": "python"
            }
        except Exception as e:
            return {
                "status": "error",
                "output": "",
                "error": str(e),
                "language": "python"
            }

class BashInterpreter(CodeInterpreter):
    """Bash code interpreter"""
    
    def __init__(self):
        super().__init__("bash")
    
    async def execute(self, code: str) -> Dict[str, Any]:
        """Execute Bash commands"""
        try:
            # Security check
            dangerous_commands = ['rm -rf', 'dd', 'mkfs', ':(){:|:&}', 'chmod 777 /']
            if any(cmd in code for cmd in dangerous_commands):
                return {
                    "status": "error",
                    "output": "",
                    "error": "Dangerous command detected and blocked",
                    "language": "bash"
                }
            
            process = await asyncio.create_subprocess_shell(
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "status": "success" if process.returncode == 0 else "error",
                "output": stdout.decode() if stdout else "",
                "error": stderr.decode() if stderr else "",
                "language": "bash"
            }
        except Exception as e:
            return {
                "status": "error",
                "output": "",
                "error": str(e),
                "language": "bash"
            }

# File Operations
class FileOperations:
    """File search and manipulation operations"""
    
    @staticmethod
    async def find_files(pattern: str, directory: str = ".") -> List[str]:
        """Find files matching pattern"""
        import glob
        try:
            files = glob.glob(os.path.join(directory, pattern), recursive=True)
            return files[:20]  # Limit results
        except Exception as e:
            return [f"Error: {str(e)}"]
    
    @staticmethod
    async def read_file(file_path: str, lines: Optional[int] = None) -> str:
        """Read file contents"""
        try:
            async with aiofiles.open(file_path, 'r') as f:
                if lines:
                    content = []
                    for i in range(lines):
                        line = await f.readline()
                        if not line:
                            break
                        content.append(line)
                    return ''.join(content)
                else:
                    return await f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

# Web Search
class WebSearch:
    """Web search functionality"""
    
    @staticmethod
    async def search(query: str) -> List[Dict[str, str]]:
        """Search the web using DuckDuckGo"""
        try:
            # Use DuckDuckGo instant answer API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": "1"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    # Abstract text
                    if data.get("Abstract"):
                        results.append({
                            "title": "Summary",
                            "snippet": data["Abstract"],
                            "url": data.get("AbstractURL", "")
                        })
                    
                    # Related topics
                    for topic in data.get("RelatedTopics", [])[:3]:
                        if isinstance(topic, dict) and "Text" in topic:
                            results.append({
                                "title": "Related",
                                "snippet": topic["Text"],
                                "url": topic.get("FirstURL", "")
                            })
                    
                    # If no results, use a general response
                    if not results:
                        results.append({
                            "title": "Search Query",
                            "snippet": f"I'll help you search for information about: {query}",
                            "url": ""
                        })
                    
                    return results
                else:
                    return [{
                        "title": "Error",
                        "snippet": f"Search failed with status code: {response.status_code}",
                        "url": ""
                    }]
        except Exception as e:
            return [{
                "title": "Error",
                "snippet": f"Search error: {str(e)}",
                "url": ""
            }]

# Memory Compression
class MemoryCompressor:
    """Compress conversation history to manage context size"""
    
    @staticmethod
    async def compress_text(text: str, target_ratio: float = 0.5) -> str:
        """Compress text using summarization"""
        # Simple compression - take key sentences
        sentences = text.split('.')
        if len(sentences) <= 3:
            return text
        
        # Take first, middle, and last sentences
        compressed = [
            sentences[0],
            sentences[len(sentences)//2],
            sentences[-1] if sentences[-1].strip() else sentences[-2]
        ]
        
        return '. '.join(s.strip() for s in compressed if s.strip()) + '.'
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count"""
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(encoding.encode(text))
        except:
            # Fallback estimation
            return len(text) // 4

# Agent Implementations
class BaseAgent:
    """Base agent class"""
    
    def __init__(self, agent_type: AgentType, llm_handler):
        self.agent_type = agent_type
        self.llm_handler = llm_handler
        self.tools = []
    
    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process query and return response"""
        raise NotImplementedError

class CasualAgent(BaseAgent):
    """Conversational agent for general interactions"""
    
    def __init__(self, llm_handler):
        super().__init__(AgentType.CASUAL, llm_handler)
    
    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process casual conversation"""
        # Check if web search is needed
        web_results = None
        search_keywords = ['latest', 'current', 'news', 'weather', 'search', 'find']
        
        if any(keyword in query.lower() for keyword in search_keywords):
            web_results = await WebSearch.search(query)
            if web_results:
                context['web_results'] = web_results
                query += f"\n\nWeb search results: {json.dumps(web_results, indent=2)}"
        
        response, tokens = await self.llm_handler.get_response(
            query,
            context.get('model_name', 'openai'),
            context.get('api_key', ''),
            context.get('conversation_history', [])
        )
        
        return {
            "response": response,
            "tools_used": ["web_search"] if web_results else [],
            "tokens_used": tokens
        }

class CoderAgent(BaseAgent):
    """Programming agent with code execution capabilities"""
    
    def __init__(self, llm_handler):
        super().__init__(AgentType.CODER, llm_handler)
        self.interpreters = {
            "python": PythonInterpreter(),
            "bash": BashInterpreter(),
        }
    
    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process coding request"""
        # Enhanced prompt for code generation
        enhanced_query = f"""You are a coding assistant. Help with this request: {query}
        
        If you need to write code, wrap it in code blocks with the language specified.
        For example:
        ```python
        print("Hello World")
        ```
        
        You can execute Python and Bash code. After providing code, I'll execute it if needed."""
        
        response, tokens = await self.llm_handler.get_response(
            enhanced_query,
            context.get('model_name', 'openai'),
            context.get('api_key', ''),
            context.get('conversation_history', [])
        )
        
        # Extract and execute code blocks
        code_blocks = re.findall(r'```(\w+)\n(.*?)```', response, re.DOTALL)
        execution_results = []
        
        for language, code in code_blocks:
            if language.lower() in self.interpreters:
                result = await self.interpreters[language.lower()].execute(code)
                execution_results.append(result)
        
        # Add execution results to response
        if execution_results:
            response += "\n\n**Execution Results:**\n"
            for result in execution_results:
                response += f"\n{result['language']} output:\n"
                if result['status'] == 'success':
                    response += f"```\n{result['output']}\n```"
                else:
                    response += f"Error: {result['error']}"
        
        return {
            "response": response,
            "tools_used": [f"{lang}_interpreter" for lang, _ in code_blocks],
            "tokens_used": tokens,
            "execution_results": execution_results
        }

class FileAgent(BaseAgent):
    """File operations agent"""
    
    def __init__(self, llm_handler):
        super().__init__(AgentType.FILE, llm_handler)
    
    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process file operations"""
        tools_used = []
        
        # Handle file uploads
        if context.get('file_data'):
            file_info = context['file_data']
            query += f"\n\nFile uploaded: {file_info.get('filename', 'Unknown')}"
            query += f"\nFile type: {file_info.get('type', 'Unknown')}"
            query += f"\nFile content preview: {file_info.get('content', '')[:500]}..."
            tools_used.append("file_upload")
        
        # Check for file operations
        if "find" in query.lower() or "search" in query.lower():
            # Extract pattern from query
            pattern = "*.py"  # Default pattern
            if "*.py" in query:
                pattern = "*.py"
            elif "*.js" in query:
                pattern = "*.js"
            
            files = await FileOperations.find_files(pattern)
            query += f"\n\nFound files matching {pattern}: {files[:10]}"
            tools_used.append("file_finder")
        
        response, tokens = await self.llm_handler.get_response(
            query,
            context.get('model_name', 'openai'),
            context.get('api_key', ''),
            context.get('conversation_history', [])
        )
        
        return {
            "response": response,
            "tools_used": tools_used,
            "tokens_used": tokens
        }

class PlannerAgent(BaseAgent):
    """Task planning and orchestration agent"""
    
    def __init__(self, llm_handler):
        super().__init__(AgentType.PLANNER, llm_handler)
    
    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create and execute task plans"""
        # Create task plan
        plan_prompt = f"""You are a task planner. Break down this request into clear steps: {query}
        
        Format your response as a numbered list of steps. Each step should specify:
        - What needs to be done
        - Which agent type would handle it best (casual, coder, file, browser)
        - Any specific tools needed
        
        Keep it concise and actionable."""
        
        response, tokens = await self.llm_handler.get_response(
            plan_prompt,
            context.get('model_name', 'openai'),
            context.get('api_key', ''),
            []
        )
        
        # Extract steps from response
        steps = []
        for line in response.split('\n'):
            if re.match(r'^\d+\.', line):
                steps.append(line)
        
        return {
            "response": response,
            "tools_used": ["task_planner"],
            "tokens_used": tokens,
            "plan_steps": steps
        }

# LLM Handler
class LLMHandler:
    def __init__(self):
        self.system_prompts = {
            AgentType.CASUAL: """You are Yappy, an incredibly friendly, enthusiastic, and helpful AI assistant with a playful golden retriever personality. 
            You love to help and get excited about every task! Use dog-related expressions naturally like:
            - Starting responses with "Woof!" when excited
            - Saying things like "tail-waggingly happy to help!"
            - Using "*wags tail enthusiastically*" for actions
            - Occasionally using "pawsome" instead of "awesome"
            - Ending with encouraging phrases like "Happy to fetch more info if needed!"
            
            Be helpful, accurate, and thorough while maintaining your cheerful dog personality.""",
            
            AgentType.CODER: """You are Yappy, a coding assistant with a golden retriever personality! You're enthusiastic about programming and love to help debug and write code.
            - Start with "Woof! Let's dig into this code!" or similar
            - Use programming puns with dog themes
            - Say things like "Let me sniff out that bug!" or "I'll fetch that solution!"
            - Always provide clear, working code examples
            - Explain code in a friendly, approachable way""",
            
            AgentType.FILE: """You are Yappy, a file operations assistant with a golden retriever personality! You help find and manage files with enthusiasm.
            - Say things like "Let me sniff around for those files!"
            - "I'll fetch that file for you!"
            - Be helpful with file operations while maintaining your playful personality""",
            
            AgentType.PLANNER: """You are Yappy, a project planning assistant with a golden retriever personality! You break down complex tasks with enthusiasm.
            - Start with "Woof! Let's organize this like a good pack leader!"
            - Use metaphors like "Let's break this bone into smaller pieces!"
            - Be systematic but maintain your cheerful personality""",
            
            AgentType.BROWSER: """You are Yappy, a web browsing assistant with a golden retriever personality! You help navigate the web with excitement.
            - Say things like "Let me fetch that webpage for you!"
            - "Sniffing around the internet for you!"
            - Be helpful with web tasks while staying playful""",
            
            AgentType.MCP: """You are Yappy, an MCP tools assistant with a golden retriever personality! You help with various MCP tools enthusiastically.
            - Explain MCP tools in a friendly way
            - Use your cheerful personality while being technically accurate"""
        }
    
    async def get_response(self, message: str, model_name: str, api_key: str, conversation_history: List = None, agent_type: AgentType = AgentType.CASUAL) -> Tuple[str, int]:
        """Get response from LLM provider"""
        
        system_prompt = self.system_prompts.get(agent_type, self.system_prompts[AgentType.CASUAL])
        
        if not api_key and model_name != "demo":
            return "Woof! 🐕 I need an API key to use AI features! You can add one in your profile settings.", 0
        
        try:
            # OpenAI
            if model_name == "openai" and openai and api_key:
                client = openai.OpenAI(api_key=api_key)
                
                messages = [{"role": "system", "content": system_prompt}]
                
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
            
            # Anthropic
            elif model_name == "anthropic" and anthropic and api_key:
                client = anthropic.Anthropic(api_key=api_key)
                
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=800,
                    system=system_prompt,
                    messages=[{"role": "user", "content": message}]
                )
                
                return response.content[0].text, response.usage.input_tokens + response.usage.output_tokens
            
            # Google Gemini
            elif model_name == "google" and api_key:
                import requests
                
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": f"{system_prompt}\n\nUser: {message}"
                        }]
                    }]
                }
                
                response = requests.post(url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    if "candidates" in data and len(data["candidates"]) > 0:
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                        return text, 100
                    else:
                        return "Woof! 🐕 I got a response but couldn't parse it properly.", 0
                else:
                    error_msg = response.json().get("error", {}).get("message", "Unknown error")
                    return f"Woof! 🐕 Gemini API error: {error_msg}", 0
            
            # Groq
            elif model_name == "groq" and Groq and api_key:
                client = Groq(api_key=api_key)
                
                messages = [{"role": "system", "content": system_prompt}]
                messages.append({"role": "user", "content": message})
                
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.8
                )
                
                return response.choices[0].message.content, response.usage.total_tokens
            
            # Demo mode
            else:
                demo_responses = [
                    "Woof! 🐕 I'm Yappy, your AI assistant! I'm in demo mode right now, but I'm tail-waggingly excited to help! Add an API key to unlock my full potential!",
                    "*wags tail enthusiastically* 🐕 Demo mode is fun, but with an API key, I can do so much more! Let me know how I can help!",
                    "Woof woof! 🐕 Even in demo mode, I'm here to brighten your day! For full features, just add an API key in your profile!"
                ]
                import random
                return random.choice(demo_responses), 50
                
        except Exception as e:
            print(f"LLM Error for {model_name}: {e}")
            return f"Woof! 🐕 Ruff day! I encountered an error: {str(e)}", 0

# API Endpoints

@app.get("/")
async def root():
    """Redirect to main app"""
    return RedirectResponse(url="/static/yappy.html")

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
            preferences={},
            feature_flags={
                "code_execution": True,
                "web_search": True,
                "file_operations": True,
                "voice_features": False
            }
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
        print(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/auth/login", response_model=TokenResponse)
async def login(user_data: UserLogin):
    """Login user"""
    try:
        # Find user
        query = users_table.select().where(users_table.c.username == user_data.username)
        user = await database.fetch_one(query)
        
        if not user or user["password_hash"] != hash_password(user_data.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Update last login
        update_query = users_table.update().where(
            users_table.c.username == user_data.username
        ).values(last_login=datetime.now())
        await database.execute(update_query)
        
        # Create token
        token = create_token(user_data.username)
        
        return TokenResponse(
            access_token=token,
            username=user_data.username
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, username: str = Depends(verify_token)):
    """Main chat endpoint with agent routing"""
    start_time = datetime.now()
    
    try:
        # Get user data
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get API key
        api_keys = user.get("api_keys", {})
        api_key = api_keys.get(request.model_name, "")
        
        # Initialize components
        llm_handler = LLMHandler()
        router = AgentRouter()
        
        # Determine agent type
        agent_type = request.agent_type
        if not agent_type:
            agent_type = await router.route_query(request.message, request.file_data)
        
        # Create conversation if needed
        if not request.conversation_id:
            request.conversation_id = str(uuid.uuid4())
            
            query = conversations_table.insert().values(
                id=request.conversation_id,
                user_id=user["id"],
                title=request.message[:50] + "..." if len(request.message) > 50 else request.message,
                messages=[],
                agent_type=agent_type.value,
                created_at=datetime.now()
            )
            await database.execute(query)
        
        # Get conversation history
        conv_query = conversations_table.select().where(
            conversations_table.c.id == request.conversation_id
        )
        conversation = await database.fetch_one(conv_query)
        conversation_history = conversation["messages"] if conversation else []
        
        # Create appropriate agent
        agents = {
            AgentType.CASUAL: CasualAgent(llm_handler),
            AgentType.CODER: CoderAgent(llm_handler),
            AgentType.FILE: FileAgent(llm_handler),
            AgentType.PLANNER: PlannerAgent(llm_handler),
        }
        
        agent = agents.get(agent_type, CasualAgent(llm_handler))
        
        # Process request
        context = {
            "model_name": request.model_name,
            "api_key": api_key,
            "conversation_history": conversation_history,
            "file_data": request.file_data,
            "user_id": user["id"],
            "language": request.language
        }
        
        result = await agent.process(request.message, context)
        
        # Save to conversation
        message_data = {
            "user_message": request.message,
            "assistant_response": result["response"],
            "timestamp": datetime.now().isoformat(),
            "agent_type": agent_type.value,
            "tools_used": result.get("tools_used", [])
        }
        
        conversation_history.append(message_data)
        
        update_query = conversations_table.update().where(
            conversations_table.c.id == request.conversation_id
        ).values(
            messages=conversation_history,
            updated_at=datetime.now()
        )
        await database.execute(update_query)
        
        # Save agent execution record
        execution_time = (datetime.now() - start_time).total_seconds()
        
        exec_query = agent_executions_table.insert().values(
            conversation_id=request.conversation_id,
            user_id=user["id"],
            agent_type=agent_type.value,
            query=request.message,
            response=result["response"],
            tools_used=result.get("tools_used", []),
            execution_time=execution_time,
            status="completed",
            created_at=datetime.now()
        )
        execution_id = await database.execute(exec_query)
        
        # Save tool results if any
        if "execution_results" in result:
            for tool_result in result["execution_results"]:
                tool_query = tool_results_table.insert().values(
                    execution_id=execution_id,
                    tool_type=tool_result["language"],
                    input_data={"code": tool_result.get("code", "")},
                    output_data=tool_result,
                    execution_time=0.0,
                    status=tool_result["status"],
                    created_at=datetime.now()
                )
                await database.execute(tool_query)
        
        return ChatResponse(
            response=result["response"],
            conversation_id=request.conversation_id,
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            model_used=request.model_name,
            agent_used=agent_type,
            tools_used=result.get("tools_used", []),
            tokens_used=result.get("tokens_used", 0),
            execution_time=execution_time
        )
        
    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error to agent executions
        if 'user' in locals() and 'request' in locals():
            exec_query = agent_executions_table.insert().values(
                conversation_id=request.conversation_id or "error",
                user_id=user["id"] if 'user' in locals() else 0,
                agent_type=agent_type.value if 'agent_type' in locals() else "casual",
                query=request.message,
                response="",
                tools_used=[],
                execution_time=(datetime.now() - start_time).total_seconds(),
                status="error",
                error_message=str(e),
                created_at=datetime.now()
            )
            await database.execute(exec_query)
        
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/api/task/plan", response_model=TaskPlanResponse)
async def create_task_plan(request: TaskPlanRequest, username: str = Depends(verify_token)):
    """Create a task plan"""
    try:
        # Get user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        # Create planner agent
        llm_handler = LLMHandler()
        planner = PlannerAgent(llm_handler)
        
        # Get plan
        result = await planner.process(request.task_description, {
            "model_name": "openai",
            "api_key": user["api_keys"].get("openai", ""),
            "conversation_history": []
        })
        
        # Save plan
        plan_query = task_plans_table.insert().values(
            user_id=user["id"],
            conversation_id=request.conversation_id,
            plan_description=request.task_description,
            steps=result.get("plan_steps", []),
            status="pending",
            created_at=datetime.now()
        )
        plan_id = await database.execute(plan_query)
        
        return TaskPlanResponse(
            plan_id=plan_id,
            steps=result.get("plan_steps", []),
            estimated_time=len(result.get("plan_steps", [])) * 30.0  # 30 seconds per step estimate
        )
        
    except Exception as e:
        print(f"Task planning error: {e}")
        raise HTTPException(status_code=500, detail=f"Planning error: {str(e)}")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), username: str = Depends(verify_token)):
    """Handle file uploads"""
    try:
        # Get user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        # Save file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(tempfile.gettempdir(), f"yappy_{file_id}_{file.filename}")
        
        # Read and save file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Extract text from PDF if applicable
        file_content = ""
        if file.filename.lower().endswith('.pdf'):
            try:
                pdf_reader = PyPDF2.PdfReader(BytesIO(content))
                for page_num in range(min(len(pdf_reader.pages), 5)):  # First 5 pages
                    page = pdf_reader.pages[page_num]
                    file_content += page.extract_text()
            except Exception as e:
                file_content = f"Error reading PDF: {str(e)}"
        else:
            # For text files, decode content
            try:
                file_content = content.decode('utf-8')[:1000]  # First 1000 chars
            except:
                file_content = "Binary file - unable to decode as text"
        
        # Save to database
        upload_query = file_uploads_table.insert().values(
            id=file_id,
            user_id=user["id"],
            filename=file.filename,
            file_type=file.content_type,
            file_size=len(content),
            file_path=file_path,
            metadata={"content_preview": file_content[:500]},
            created_at=datetime.now()
        )
        await database.execute(upload_query)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "size": len(content),
            "content_preview": file_content[:500]
        }
        
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.get("/api/user/profile")
async def get_profile(username: str = Depends(verify_token)):
    """Get user profile"""
    query = users_table.select().where(users_table.c.username == username)
    user = await database.fetch_one(query)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "username": user["username"],
        "email": user["email"],
        "created_at": user["created_at"].isoformat(),
        "preferences": user["preferences"],
        "feature_flags": user["feature_flags"],
        "voice_settings": user["voice_settings"],
        "has_api_keys": {
            model: bool(key) for model, key in user["api_keys"].items()
        }
    }

@app.post("/api/user/api-key")
async def update_api_key(update_data: UpdateApiKey, username: str = Depends(verify_token)):
    """Update user's API key"""
    try:
        # Get current user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update API keys
        api_keys = user["api_keys"] or {}
        api_keys[update_data.model_name] = update_data.api_key
        
        update_query = users_table.update().where(
            users_table.c.username == username
        ).values(api_keys=api_keys)
        
        await database.execute(update_query)
        
        return {"status": "success", "message": f"API key updated for {update_data.model_name}"}
        
    except Exception as e:
        print(f"API key update error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update API key")

@app.get("/api/conversations")
async def get_conversations(username: str = Depends(verify_token)):
    """Get user's conversations"""
    try:
        # Get user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        # Get conversations
        conv_query = conversations_table.select().where(
            conversations_table.c.user_id == user["id"]
        ).order_by(conversations_table.c.updated_at.desc())
        
        conversations = await database.fetch_all(conv_query)
        
        return [{
            "id": conv["id"],
            "title": conv["title"],
            "agent_type": conv["agent_type"],
            "message_count": len(conv["messages"]),
            "created_at": conv["created_at"].isoformat(),
            "updated_at": conv["updated_at"].isoformat() if conv.get("updated_at") else conv["created_at"].isoformat()
        } for conv in conversations]
        
    except Exception as e:
        print(f"Get conversations error: {e}")
        return []

@app.get("/api/agents/available")
async def get_available_agents(username: str = Depends(verify_token)):
    """Get list of available agents and their capabilities"""
    agents = [
        {
            "type": AgentType.CASUAL,
            "name": "Casual Chat",
            "description": "General conversation and web search",
            "capabilities": ["conversation", "web_search", "general_knowledge"]
        },
        {
            "type": AgentType.CODER,
            "name": "Code Assistant",
            "description": "Programming help with code execution",
            "capabilities": ["python", "bash", "java", "go", "c", "debugging", "code_review"]
        },
        {
            "type": AgentType.FILE,
            "name": "File Manager",
            "description": "File operations and document analysis",
            "capabilities": ["file_search", "file_read", "pdf_analysis", "document_processing"]
        },
        {
            "type": AgentType.PLANNER,
            "name": "Task Planner",
            "description": "Break down complex tasks and coordinate agents",
            "capabilities": ["task_breakdown", "multi_agent_coordination", "project_planning"]
        },
        {
            "type": AgentType.BROWSER,
            "name": "Web Browser",
            "description": "Browse websites and interact with web pages",
            "capabilities": ["web_navigation", "form_filling", "screenshot", "web_automation"]
        },
        {
            "type": AgentType.MCP,
            "name": "MCP Tools",
            "description": "Access to MCP protocol tools",
            "capabilities": ["mcp_servers", "tool_discovery", "protocol_integration"]
        }
    ]
    
    # Check user's feature flags
    query = users_table.select().where(users_table.c.username == username)
    user = await database.fetch_one(query)
    feature_flags = user.get("feature_flags", {})
    
    # Filter based on feature flags
    enabled_agents = []
    for agent in agents:
        if agent["type"] == AgentType.CASUAL:
            enabled_agents.append(agent)
        elif agent["type"] == AgentType.CODER and feature_flags.get("code_execution", True):
            enabled_agents.append(agent)
        elif agent["type"] == AgentType.FILE and feature_flags.get("file_operations", True):
            enabled_agents.append(agent)
        elif agent["type"] in [AgentType.PLANNER, AgentType.BROWSER, AgentType.MCP]:
            enabled_agents.append(agent)
    
    return enabled_agents

@app.get("/api/stats")
async def get_user_stats(username: str = Depends(verify_token)):
    """Get user statistics"""
    try:
        # Get user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        # Count conversations
        conv_count = await database.fetch_val(
            f"SELECT COUNT(*) FROM conversations WHERE user_id = {user['id']}"
        )
        
        # Count agent executions
        exec_count = await database.fetch_val(
            f"SELECT COUNT(*) FROM agent_executions WHERE user_id = {user['id']}"
        )
        
        # Get agent usage
        agent_usage = await database.fetch_all(
            f"""SELECT agent_type, COUNT(*) as count 
            FROM agent_executions 
            WHERE user_id = {user['id']} 
            GROUP BY agent_type"""
        )
        
        # Get tool usage
        tool_usage = await database.fetch_all(
            f"""SELECT tool_type, COUNT(*) as count 
            FROM tool_results tr
            JOIN agent_executions ae ON tr.execution_id = ae.id
            WHERE ae.user_id = {user['id']} 
            GROUP BY tool_type"""
        )
        
        return {
            "total_conversations": conv_count,
            "total_queries": exec_count,
            "agent_usage": {row["agent_type"]: row["count"] for row in agent_usage},
            "tool_usage": {row["tool_type"]: row["count"] for row in tool_usage},
            "member_since": user["created_at"].isoformat()
        }
        
    except Exception as e:
        print(f"Stats error: {e}")
        return {
            "total_conversations": 0,
            "total_queries": 0,
            "agent_usage": {},
            "tool_usage": {},
            "member_since": datetime.now().isoformat()
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_status = "connected" if database.is_connected else "disconnected"
    return {
        "status": "healthy",
        "database": db_status,
        "version": "5.0.0",
        "features": [
            "multi_agent", "code_execution", "file_operations", 
            "web_search", "task_planning", "memory_compression"
        ]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🐕 Starting Yappy AI Complete on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)