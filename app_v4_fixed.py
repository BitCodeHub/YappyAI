import os
import json
import uuid
import hashlib
import secrets
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from databases import Database
import sqlalchemy
from sqlalchemy import MetaData, Table, Column, Integer, String, DateTime, JSON
import httpx
import re

# Import LLM libraries
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from groq import Groq
except ImportError:
    Groq = None

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./yappy_v4.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

database = Database(DATABASE_URL)
metadata = MetaData()

# Database Tables
users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(50), unique=True, index=True, nullable=False),
    Column("email", String(100)),
    Column("password_hash", String(255), nullable=False),
    Column("api_keys", JSON, default={}),
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
    Column("updated_at", DateTime, default=datetime.now),
)

engine = sqlalchemy.create_engine(DATABASE_URL)

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🐕 Yappy AI v4 - Starting up... Current date: {datetime.now().strftime('%B %d, %Y')}")
    try:
        await database.connect()
        metadata.create_all(bind=engine)
        print("✅ Database connected")
    except Exception as e:
        print(f"❌ Database error: {e}")
    yield
    await database.disconnect()

# Initialize FastAPI
app = FastAPI(
    title="Yappy AI v4 Fixed",
    description="AI with working context and real-time data",
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

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Security
security = HTTPBearer(auto_error=False)

# Models
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

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    timestamp: str
    model_used: str
    web_searched: bool = False
    context_used: bool = False

# Helpers
def hash_password(password: str) -> str:
    return hashlib.sha256(f"{password}yappy2024".encode()).hexdigest()

def create_token(username: str) -> str:
    return f"{username}:{secrets.token_urlsafe(32)}"

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        username = credentials.credentials.split(":")[0]
        return username
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# Enhanced Web Search with Context Awareness
class ContextAwareSearch:
    @staticmethod
    def extract_context_from_history(messages: List[Dict], current_query: str) -> Dict[str, Any]:
        """Extract relevant context from conversation history"""
        context = {
            "previous_topic": None,
            "location": None,
            "date_context": None,
            "entities": []
        }
        
        # Look at last few messages
        recent_messages = messages[-3:] if len(messages) >= 3 else messages
        
        for msg in recent_messages:
            user_msg = msg.get("user_message", "").lower()
            assistant_msg = msg.get("assistant_response", "").lower()
            
            # Extract location
            location_patterns = [
                r'weather (?:in|at|for) ([\w\s,]+)',
                r'in ([\w\s,]+)',
                r'at ([\w\s,]+)'
            ]
            
            for pattern in location_patterns:
                loc_match = re.search(pattern, user_msg, re.IGNORECASE)
                if loc_match:
                    context["location"] = loc_match.group(1).strip()
                    break
            
            # Extract topic
            if "weather" in user_msg or "weather" in assistant_msg:
                context["previous_topic"] = "weather"
            elif "nba" in user_msg or "basketball" in assistant_msg:
                context["previous_topic"] = "nba"
            elif "forecast" in user_msg or "forecast" in assistant_msg:
                context["previous_topic"] = "forecast"
        
        # Check if current query is a follow-up
        follow_up_indicators = ["next", "7 days", "forecast", "more", "what about", "how about", "and"]
        current_lower = current_query.lower()
        
        for indicator in follow_up_indicators:
            if indicator in current_lower:
                return context
        
        return context
    
    @staticmethod
    async def search_with_context(query: str, context: Dict[str, Any]) -> str:
        """Perform search considering conversation context"""
        search_info = ""
        
        # Handle follow-up queries
        if context.get("previous_topic") == "weather" and context.get("location"):
            # This is a weather follow-up
            if "7 days" in query.lower() or "forecast" in query.lower() or "next" in query.lower():
                location = context["location"]
                
                try:
                    async with httpx.AsyncClient() as client:
                        # Get 7-day forecast
                        forecast_response = await client.get(
                            f"https://wttr.in/{location}?format=j1",
                            timeout=5.0
                        )
                        
                        if forecast_response.status_code == 200:
                            data = forecast_response.json()
                            search_info = f"7-Day Weather Forecast for {location}:\n\n"
                            
                            # Current date
                            current_date = datetime.now()
                            search_info += f"(Current date: {current_date.strftime('%A, %B %d, %Y')})\n\n"
                            
                            # Get forecast
                            weather = data.get("weather", [])
                            for i, day in enumerate(weather[:7]):
                                date = day.get("date", "")
                                max_temp_f = day.get("maxtempF", "N/A")
                                min_temp_f = day.get("mintempF", "N/A")
                                
                                # Get weather description
                                hourly = day.get("hourly", [{}])
                                if hourly:
                                    desc = hourly[4].get("weatherDesc", [{}])[0].get("value", "N/A")
                                else:
                                    desc = "N/A"
                                
                                # Format day name
                                try:
                                    date_obj = datetime.strptime(date, "%Y-%m-%d")
                                    day_name = date_obj.strftime("%A, %B %d")
                                except:
                                    day_name = f"Day {i+1}"
                                
                                search_info += f"{day_name}:\n"
                                search_info += f"  High: {max_temp_f}°F, Low: {min_temp_f}°F\n"
                                search_info += f"  Conditions: {desc}\n\n"
                            
                            return search_info
                except Exception as e:
                    print(f"Forecast error: {e}")
        
        # Regular search for non-context queries
        return await ContextAwareSearch.regular_search(query)
    
    @staticmethod
    async def regular_search(query: str) -> str:
        """Regular search without context"""
        try:
            async with httpx.AsyncClient() as client:
                search_info = ""
                
                # Weather queries
                if "weather" in query.lower():
                    location = "New York"
                    location_match = re.search(r'weather (?:in|at|for) ([\w\s,]+)', query, re.IGNORECASE)
                    if location_match:
                        location = location_match.group(1).strip()
                    
                    weather_response = await client.get(
                        f"https://wttr.in/{location}?format=j1",
                        timeout=5.0
                    )
                    
                    if weather_response.status_code == 200:
                        data = weather_response.json()
                        current = data.get("current_condition", [{}])[0]
                        
                        search_info = f"Current Weather in {location}:\n"
                        search_info += f"Temperature: {current.get('temp_F', 'N/A')}°F ({current.get('temp_C', 'N/A')}°C)\n"
                        search_info += f"Feels like: {current.get('FeelsLikeF', 'N/A')}°F\n"
                        search_info += f"Condition: {current.get('weatherDesc', [{}])[0].get('value', 'N/A')}\n"
                        search_info += f"Humidity: {current.get('humidity', 'N/A')}%\n"
                        search_info += f"Wind: {current.get('windspeedMiles', 'N/A')} mph\n"
                        search_info += f"\n(Data retrieved: {datetime.now().strftime('%B %d, %Y at %I:%M %p')})"
                
                # NBA queries
                elif any(term in query.lower() for term in ['nba', 'basketball', 'game', 'playing']):
                    nba_response = await client.get(
                        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
                        timeout=5.0
                    )
                    
                    if nba_response.status_code == 200:
                        data = nba_response.json()
                        search_info = f"NBA Games Today ({datetime.now().strftime('%B %d, %Y')}):\n\n"
                        
                        events = data.get("events", [])
                        if not events:
                            search_info += "No NBA games scheduled for today.\n\n"
                            
                            # Get upcoming games
                            search_info += "Upcoming NBA Schedule:\n"
                            # You could make another API call for schedule here
                        else:
                            for event in events[:10]:
                                name = event.get("name", "")
                                status = event.get("status", {}).get("type", {}).get("description", "")
                                
                                # Get game time
                                date_str = event.get("date", "")
                                
                                competitions = event.get("competitions", [{}])[0]
                                competitors = competitions.get("competitors", [])
                                
                                if len(competitors) >= 2:
                                    team1 = competitors[0]
                                    team2 = competitors[1]
                                    
                                    team1_name = team1.get("team", {}).get("displayName", "")
                                    team1_score = team1.get("score", "0")
                                    team2_name = team2.get("team", {}).get("displayName", "")
                                    team2_score = team2.get("score", "0")
                                    
                                    search_info += f"{team1_name} vs {team2_name}\n"
                                    
                                    if status == "Final":
                                        search_info += f"Final Score: {team1_name} {team1_score} - {team2_name} {team2_score}\n"
                                        # Determine winner
                                        if int(team1_score) > int(team2_score):
                                            search_info += f"Winner: {team1_name}\n"
                                        else:
                                            search_info += f"Winner: {team2_name}\n"
                                    elif "In Progress" in status:
                                        search_info += f"Live: {team1_score} - {team2_score} ({status})\n"
                                    else:
                                        search_info += f"Status: {status}\n"
                                    
                                    search_info += "\n"
                
                # General web search fallback
                if not search_info:
                    ddg_response = await client.get(
                        "https://api.duckduckgo.com/",
                        params={"q": query, "format": "json", "no_html": "1"},
                        timeout=5.0
                    )
                    
                    if ddg_response.status_code == 200:
                        data = ddg_response.json()
                        if data.get("Abstract"):
                            search_info = f"Information found:\n{data['Abstract']}\n"
                        elif data.get("Answer"):
                            search_info = f"Answer: {data['Answer']}\n"
                
                return search_info if search_info else "No specific results found."
                
        except Exception as e:
            print(f"Search error: {e}")
            return f"Search error: {str(e)}"

# LLM Handler with Context
class LLMHandler:
    def __init__(self):
        # Update system prompt with current date
        current_date = datetime.now().strftime("%B %d, %Y")
        self.system_prompt = f"""You are Yappy, a helpful AI assistant. Today's date is {current_date}.

IMPORTANT INSTRUCTIONS:
1. When users ask follow-up questions (like "next 7 days" after asking about weather), understand they're referring to the previous topic
2. Always use the current date ({current_date}) for any date-related responses
3. When you receive search results, use them as the primary source of truth
4. For weather forecasts, provide specific dates and day names
5. For NBA games, mention if games are live, final scores, or upcoming

Be friendly and helpful while providing accurate, current information."""
    
    async def get_response(self, message: str, search_info: str, conversation_history: List[Dict], model_name: str, api_key: str) -> Tuple[str, int]:
        """Get response with context and search info"""
        
        # Build conversation context
        context_messages = []
        for msg in conversation_history[-5:]:  # Last 5 exchanges
            if "user_message" in msg:
                context_messages.append(f"User: {msg['user_message']}")
            if "assistant_response" in msg:
                context_messages.append(f"Assistant: {msg['assistant_response']}")
        
        # Create enhanced message
        enhanced_message = ""
        
        if context_messages:
            enhanced_message += "Recent conversation:\n"
            enhanced_message += "\n".join(context_messages[-4:])  # Last 2 exchanges
            enhanced_message += "\n\n"
        
        if search_info and search_info != "No specific results found.":
            enhanced_message += f"Current search results:\n{search_info}\n\n"
        
        enhanced_message += f"Current user question: {message}"
        
        if not api_key and model_name != "demo":
            return "Woof! 🐕 I need an API key to provide AI responses. Please add one in your profile!", 0
        
        try:
            # OpenAI
            if model_name == "openai" and openai and api_key:
                client = openai.OpenAI(api_key=api_key)
                
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": enhanced_message}
                ]
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.7
                )
                
                return response.choices[0].message.content, response.usage.total_tokens
            
            # Anthropic
            elif model_name == "anthropic" and anthropic and api_key:
                client = anthropic.Anthropic(api_key=api_key)
                
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=800,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": enhanced_message}]
                )
                
                return response.content[0].text, response.usage.input_tokens + response.usage.output_tokens
            
            # Demo mode
            else:
                if search_info and search_info != "No specific results found.":
                    return f"Woof! 🐕 Based on the search:\n\n{search_info}\n\nFor full AI features, add an API key!", 50
                else:
                    return "Woof! 🐕 I'm in demo mode. Add an API key for full features!", 50
                
        except Exception as e:
            return f"Error: {str(e)}", 0

# API Endpoints

@app.get("/")
async def root():
    return RedirectResponse(url="/static/yappy_v4.html")

@app.post("/auth/register")
async def register(user_data: UserSignup):
    try:
        query = users_table.select().where(users_table.c.username == user_data.username)
        existing = await database.fetch_one(query)
        
        if existing:
            raise HTTPException(status_code=400, detail="Username exists")
        
        query = users_table.insert().values(
            username=user_data.username,
            email=user_data.email,
            password_hash=hash_password(user_data.password),
            created_at=datetime.now(),
            api_keys={}
        )
        
        await database.execute(query)
        
        return {
            "access_token": create_token(user_data.username),
            "token_type": "bearer",
            "username": user_data.username
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/login")
async def login(user_data: UserLogin):
    query = users_table.select().where(users_table.c.username == user_data.username)
    user = await database.fetch_one(query)
    
    if not user or user["password_hash"] != hash_password(user_data.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {
        "access_token": create_token(user_data.username),
        "token_type": "bearer",
        "username": user_data.username
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, username: str = Depends(verify_token)):
    """Chat with context awareness and web search"""
    try:
        # Get user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get API key
        api_key = user.get("api_keys", {}).get(request.model_name, "")
        
        # Get conversation history
        conversation_history = []
        if request.conversation_id:
            conv_query = conversations_table.select().where(
                conversations_table.c.id == request.conversation_id
            )
            conversation = await database.fetch_one(conv_query)
            if conversation:
                conversation_history = conversation.get("messages", [])
        
        # Extract context from history
        context = ContextAwareSearch.extract_context_from_history(
            conversation_history, 
            request.message
        )
        
        # Determine if we need search
        needs_search = False
        context_used = False
        
        # Always search for questions or specific patterns
        if '?' in request.message:
            needs_search = True
        
        # Check for follow-up queries
        if context.get("previous_topic") and any(
            indicator in request.message.lower() 
            for indicator in ["next", "7 days", "forecast", "more", "what about"]
        ):
            needs_search = True
            context_used = True
        
        # Search for specific topics
        search_triggers = [
            'weather', 'nba', 'basketball', 'game', 'score',
            'who', 'what', 'when', 'where', 'how',
            'current', 'latest', 'today', 'now'
        ]
        
        if any(trigger in request.message.lower() for trigger in search_triggers):
            needs_search = True
        
        # Perform search
        search_info = ""
        if needs_search:
            if context_used:
                search_info = await ContextAwareSearch.search_with_context(
                    request.message, 
                    context
                )
            else:
                search_info = await ContextAwareSearch.regular_search(request.message)
        
        # Get LLM response
        llm_handler = LLMHandler()
        response_text, tokens = await llm_handler.get_response(
            request.message,
            search_info,
            conversation_history,
            request.model_name,
            api_key
        )
        
        # Create/update conversation
        if not request.conversation_id:
            request.conversation_id = str(uuid.uuid4())
            
            conv_query = conversations_table.insert().values(
                id=request.conversation_id,
                user_id=user["id"],
                title=request.message[:50],
                messages=[],
                created_at=datetime.now()
            )
            await database.execute(conv_query)
        
        # Save message
        conversation_history.append({
            "user_message": request.message,
            "assistant_response": response_text,
            "timestamp": datetime.now().isoformat(),
            "web_searched": needs_search,
            "context_used": context_used,
            "search_info": search_info if needs_search else None
        })
        
        update_query = conversations_table.update().where(
            conversations_table.c.id == request.conversation_id
        ).values(
            messages=conversation_history,
            updated_at=datetime.now()
        )
        await database.execute(update_query)
        
        return ChatResponse(
            response=response_text,
            conversation_id=request.conversation_id,
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            model_used=request.model_name,
            web_searched=needs_search,
            context_used=context_used
        )
        
    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/user/api-key")
async def update_api_key(update_data: UpdateApiKey, username: str = Depends(verify_token)):
    query = users_table.select().where(users_table.c.username == username)
    user = await database.fetch_one(query)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    api_keys = user["api_keys"] or {}
    api_keys[update_data.model_name] = update_data.api_key
    
    update_query = users_table.update().where(
        users_table.c.username == username
    ).values(api_keys=api_keys)
    
    await database.execute(update_query)
    
    return {"status": "success"}

@app.get("/api/conversations")
async def get_conversations(username: str = Depends(verify_token)):
    query = users_table.select().where(users_table.c.username == username)
    user = await database.fetch_one(query)
    
    conv_query = conversations_table.select().where(
        conversations_table.c.user_id == user["id"]
    ).order_by(conversations_table.c.updated_at.desc())
    
    conversations = await database.fetch_all(conv_query)
    
    return [{
        "id": conv["id"],
        "title": conv["title"],
        "created_at": conv["created_at"].isoformat()
    } for conv in conversations]

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "4.0.0",
        "current_date": datetime.now().strftime("%B %d, %Y"),
        "features": ["context_awareness", "real_time_search", "conversation_memory"]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🐕 Starting Yappy AI v4 Fixed on port {port}...")
    print(f"📅 Current date: {datetime.now().strftime('%B %d, %Y')}")
    uvicorn.run(app, host="0.0.0.0", port=port)