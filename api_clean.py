#!/usr/bin/env python3
"""
Clean API for Neo with working charts and no duplicates
"""

import os
import uuid
import json
import base64
import io
import warnings
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import google.generativeai as genai

# Chart libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

warnings.filterwarnings("ignore")

# Configure API - No default API key, user must provide
genai.configure(api_key="")

# LLM Provider Functions
def get_llm_response(prompt: str, model_name: str = None, api_key: str = None, image_data=None):
    """Get response from specified LLM provider"""
    
    # Require API key and model
    if not model_name or not api_key:
        return "❌ **Error**: No LLM model or API key provided. Please log out and log back in with your API key."
    
    # Create consistent system prompt for all providers
    system_prompt = """You are Yappy, a friendly AI assistant. You are helpful, knowledgeable, and conversational. 

IMPORTANT: Always identify yourself as "Yappy" when asked about your identity. Do not mention any specific company, training organization, or underlying model. You are simply "Yappy, your AI assistant."

When provided with web search results or current data, present the information as factual and current. Do not say you cannot access real-time information if the data has been provided to you through search results.

Respond naturally and helpfully to user questions."""
    
    # Combine system prompt with user prompt
    if image_data:
        full_prompt = f"{system_prompt}\n\nUser: {prompt}"
    else:
        full_prompt = f"{system_prompt}\n\nUser: {prompt}"
    
    # Handle different providers
    if model_name == "google":
        # Use provided Gemini API key
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash', system_instruction=system_prompt)
        if image_data:
            response = model.generate_content([prompt, image_data])
        else:
            response = model.generate_content(prompt)
        # Reset to empty key
        genai.configure(api_key="")
        return response.text
        
    elif model_name == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        if image_data:
            # For OpenAI, we'd need to handle image differently
            # For now, just mention it's not supported
            messages[1]["content"] += "\n\n[Note: Image analysis is not supported with OpenAI in this implementation]"
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content
        
    elif model_name == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": prompt}],
            system=system_prompt,
            max_tokens=1024
        )
        return response.content[0].text
        
    elif model_name == "groq":
        from groq import Groq
        client = Groq(api_key=api_key)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content
        
    elif model_name == "ollama":
        # For Ollama, use the API endpoint
        import requests
        url = api_key if api_key else "http://localhost:11434"
        
        # Combine system prompt with user prompt for Ollama
        combined_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nYappy:"
        
        response = requests.post(
            f"{url}/api/generate",
            json={
                "model": "llama3",
                "prompt": combined_prompt,
                "stream": False
            }
        )
        return response.json()["response"]
    
    else:
        # No fallback - require valid provider
        return f"❌ **Error**: Unsupported LLM provider '{model_name}'. Please use: google, openai, anthropic, groq, or ollama."

# Models
class FileData(BaseModel):
    name: str
    type: str
    content: str

class QueryRequest(BaseModel):
    query: str
    file_data: Optional[FileData] = None
    llm_model: Optional[str] = None
    api_key: Optional[str] = None

class QueryResponse(BaseModel):
    done: str
    answer: str
    reasoning: str
    agent_name: str
    success: str
    blocks: dict
    status: str
    uid: str

# Authentication Models
class UserCredentials(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    expires_in: int

# Initialize FastAPI
app = FastAPI(title="Yappy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    @app.get("/")
    async def serve_frontend():
        from fastapi.responses import FileResponse
        return FileResponse("static/index.html")

# Global state
last_response = None
is_generating = False
pdf_context = {}  # Store PDF content for continued Q&A
image_context = {}  # Store image data for continued Q&A
conversation_history = []  # Store conversation history for context
last_response_id = None  # Track last response ID to prevent duplicates

def get_conversation_context():
    """Build conversation context string from history"""
    if not conversation_history:
        return ""
    
    context = "\n\nConversation History:\n"
    for i, msg in enumerate(conversation_history[-6:]):  # Last 6 messages
        context += f"{msg['role']}: {msg['content']}\n"
    
    return context

def add_to_conversation(role, content):
    """Add message to conversation history"""
    global conversation_history
    conversation_history.append({
        'role': role,
        'content': content[:500]  # Limit length to prevent context explosion
    })
    
    # Keep only last 10 messages to prevent memory growth
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]

def detect_task_request(query):
    """Detect if user is asking for help with a task"""
    task_indicators = [
        'how to', 'how do i', 'how can i', 'steps to', 'guide to', 'tutorial',
        'teach me', 'show me how', 'help me', 'i want to', 'i need to',
        'process for', 'procedure to', 'method to', 'way to do',
        'instructions for', 'plan to', 'roadmap', 'strategy for'
    ]
    
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in task_indicators)

async def break_down_task(task_query, llm_model=None, api_key=None):
    """Break down a complex task into subtasks and research each one"""
    print(f"🎯 TASK BREAKDOWN MODE: {task_query}")
    
    # Step 1: Break down the main task
    breakdown_prompt = f"""You are a task planning expert. Break down this task into specific, actionable subtasks:

Task: {task_query}

Provide a structured breakdown with:
1. Clear, specific subtasks (3-8 steps)
2. Each subtask should be concrete and actionable
3. Include time estimates where possible
4. Identify which subtasks might need web research for current information

Format your response as:
**TASK BREAKDOWN: [Task Name]**

**📋 SUBTASKS:**
1. [Subtask 1] - [Time estimate] - [Research needed: YES/NO]
2. [Subtask 2] - [Time estimate] - [Research needed: YES/NO]
...

**💡 KEY CONSIDERATIONS:**
- [Important considerations]
- [Potential challenges]
- [Cost-saving tips]

**🔍 RESEARCH PRIORITIES:**
- [List subtasks that need current web information]
"""
    
    breakdown = get_llm_response(breakdown_prompt, llm_model, api_key)
    print(f"📋 TASK BROKEN DOWN: {len(breakdown)} characters")
    
    # Step 2: Identify subtasks that need web research
    research_prompt = f"""Based on this task breakdown, identify which specific subtasks need current web information:

{breakdown}

Return ONLY a numbered list of subtasks that would benefit from web research (current prices, methods, tools, regulations, etc.):
1. [Specific subtask needing research]
2. [Another subtask needing research]
..."""
    
    research_tasks = get_llm_response(research_prompt, llm_model, api_key)
    print(f"🔍 RESEARCH TASKS IDENTIFIED: {research_tasks}")
    
    # Step 3: Perform web research for identified subtasks
    researched_info = ""
    if research_tasks and len(research_tasks.strip()) > 10:
        try:
            # Check if SearxNG is available for web search
            import requests
            try:
                requests.get("http://127.0.0.1:8080", timeout=2)
                searxng_available = True
            except:
                searxng_available = False
                
            if searxng_available:
                from sources.tools.searxSearch import searxSearch
                search_tool = searxSearch(base_url="http://127.0.0.1:8080")
                
                # Research each identified subtask
                research_results = []
                research_lines = [line.strip() for line in research_tasks.split('\n') if line.strip() and not line.strip().startswith('Return')]
                
                for i, research_task in enumerate(research_lines[:3]):  # Limit to 3 searches
                    if research_task:
                        print(f"🌐 RESEARCHING: {research_task}")
                        search_results = search_tool.execute([research_task])
                        
                        if not search_tool.execution_failure_check(search_results):
                            research_results.append(f"**Research for: {research_task}**\n{search_results}\n")
                
                if research_results:
                    researched_info = "\n".join(research_results)
                    print(f"✅ WEB RESEARCH COMPLETED: {len(researched_info)} characters")
            else:
                print("📶 SearxNG not available - using AI knowledge only")
        except Exception as e:
            print(f"❌ RESEARCH ERROR: {e}")
    
    # Step 4: Compile comprehensive guide with research
    final_prompt = f"""Create a comprehensive, actionable guide by combining the task breakdown with web research results:

Original Task: {task_query}

Task Breakdown:
{breakdown}

Web Research Results:
{researched_info if researched_info else "Using AI knowledge base"}

Create a COMPLETE, ACTIONABLE GUIDE with:

**🎯 COMPLETE TASK GUIDE: [Task Name]**

**⚡ QUICK START (What to do right now):**
- [Immediate first step]
- [What you need to gather/prepare]

**📋 DETAILED STEP-BY-STEP PROCESS:**
1. **[Step Name]** - [Time: X minutes/hours]
   - Specific actions to take
   - Tools/resources needed
   - Current costs/pricing (from research)
   - Tips for efficiency

2. **[Step Name]** - [Time: X minutes/hours]
   - Specific actions to take
   - Tools/resources needed
   - Tips for success

[Continue for all steps...]

**💰 COST BREAKDOWN:**
- [Estimated costs from research]
- [Money-saving alternatives]
- [Free options available]

**⏰ TIME ESTIMATE:**
- Total time: [X hours/days]
- Can be done in: [timeframe]

**🛠️ TOOLS & RESOURCES NEEDED:**
- [List specific tools, websites, apps]
- [Where to get them]
- [Current pricing from research]

**⚠️ COMMON PITFALLS TO AVOID:**
- [Potential issues and how to prevent them]

**🏆 SUCCESS TIPS:**
- [Pro tips for better results]
- [Ways to save time and money]

**📞 WHEN TO GET HELP:**
- [Situations where professional help is worth it]

Use current information from web research where available. Make this guide immediately actionable and comprehensive."""
    
    final_guide = get_llm_response(final_prompt, llm_model, api_key)
    
    # Add research source indicator
    if researched_info:
        final_guide += "\n\n🌐 **Enhanced with current web research for accuracy and up-to-date information**"
    
    print(f"✅ COMPREHENSIVE GUIDE CREATED: {len(final_guide)} characters")
    return final_guide

# Simple authentication (for demo purposes - not secure for production)
users_db = {}  # In production, use a proper database

def create_access_token(username: str) -> str:
    """Create a simple token (in production, use JWT with expiration)"""
    import time
    return f"token_{username}_{int(time.time())}"

def get_current_user(authorization: str = Header(None)) -> str:
    """Simple token validation (in production, use proper JWT validation)"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    token = authorization.split(" ")[1]
    
    # Simple validation - in production, validate JWT
    if not token.startswith("token_"):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Extract username from token (very basic - use JWT in production)
    try:
        username = token.split("_")[1]
        return username
    except:
        raise HTTPException(status_code=401, detail="Invalid token format")

# Authentication endpoints
@app.post("/auth/register")
async def register(credentials: UserCredentials):
    """Register a new user"""
    if credentials.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Simple password storage (in production, hash passwords properly)
    users_db[credentials.username] = credentials.password
    
    return {"message": "User registered successfully", "username": credentials.username}

@app.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserCredentials):
    """Login and receive access token"""
    if credentials.username not in users_db:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    if users_db[credentials.username] != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    access_token = create_access_token(credentials.username)
    
    return TokenResponse(
        access_token=access_token,
        expires_in=86400  # 24 hours
    )

@app.get("/auth/me")
async def get_current_user_info(current_user: str = Depends(get_current_user)):
    """Get current authenticated user"""
    return {"username": current_user}

def extract_pdf_content(file_data):
    """Extract comprehensive content from PDF including text, metadata, and structure"""
    try:
        print("Starting PDF extraction...")
        print(f"File data type: {type(file_data)}")
        print(f"Content length: {len(file_data.content) if hasattr(file_data, 'content') else 'No content attr'}")
        
        import pypdf
        print("pypdf imported successfully")
        
        # Decode base64 PDF
        pdf_data = file_data.content
        if ',' in pdf_data:  # Remove data URL prefix if present
            pdf_data = pdf_data.split(',')[1]
            print("Removed data URL prefix")
        
        print(f"Decoding base64 data of length: {len(pdf_data)}")
        pdf_bytes = base64.b64decode(pdf_data)
        print(f"Decoded to {len(pdf_bytes)} bytes")
        
        pdf_file = io.BytesIO(pdf_bytes)
        
        # Create PDF reader
        print("Creating PDF reader...")
        pdf_reader = pypdf.PdfReader(pdf_file)
        print(f"PDF reader created successfully. Pages: {len(pdf_reader.pages)}")
        
        # Extract metadata
        metadata = {}
        if pdf_reader.metadata:
            metadata = {
                'title': pdf_reader.metadata.get('/Title', ''),
                'author': pdf_reader.metadata.get('/Author', ''),
                'subject': pdf_reader.metadata.get('/Subject', ''),
                'creator': pdf_reader.metadata.get('/Creator', ''),
                'producer': pdf_reader.metadata.get('/Producer', ''),
                'creation_date': str(pdf_reader.metadata.get('/CreationDate', '')),
                'modification_date': str(pdf_reader.metadata.get('/ModDate', ''))
            }
            print(f"Metadata extracted: {metadata}")
        
        # Extract text from all pages
        full_text = ""
        page_texts = []
        total_pages = len(pdf_reader.pages)
        
        print(f"Extracting text from {total_pages} pages...")
        for page_num in range(total_pages):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    print(f"Page {page_num + 1}: Extracted {len(page_text)} characters")
                else:
                    print(f"Page {page_num + 1}: No text extracted")
                    
                page_texts.append({
                    'page_number': page_num + 1,
                    'text': page_text
                })
                full_text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
            except Exception as e:
                print(f"Error extracting page {page_num + 1}: {e}")
                page_texts.append({
                    'page_number': page_num + 1,
                    'text': f"[Error extracting text from page {page_num + 1}]"
                })
        
        # Calculate statistics
        word_count = len(full_text.split())
        char_count = len(full_text)
        print(f"Total text extracted: {char_count} characters, {word_count} words")
        
        # Try to extract images count (if supported)
        image_count = 0
        try:
            for page in pdf_reader.pages:
                if hasattr(page, 'images'):
                    image_count += len(list(page.images))
        except:
            pass
        
        # Extract any forms/fields
        form_fields = []
        try:
            if hasattr(pdf_reader, 'get_form_text_fields') and pdf_reader.get_form_text_fields():
                form_fields = list(pdf_reader.get_form_text_fields().keys())
        except:
            pass
        
        # Create comprehensive content object
        pdf_content = {
            'full_text': full_text,
            'page_texts': page_texts,
            'total_pages': total_pages,
            'word_count': word_count,
            'char_count': char_count,
            'image_count': image_count,
            'metadata': metadata,
            'form_fields': form_fields,
            'has_text': bool(full_text.strip())
        }
        
        print(f"PDF content extracted successfully. Has text: {pdf_content['has_text']}")
        return pdf_content
        
    except ImportError:
        print("pypdf not installed. Installing...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
        
        # Try again after installation
        return extract_pdf_content(file_data)
        
    except Exception as e:
        print(f"PDF extraction error: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_advanced_visualizations(data, numeric_columns, request_text=""):
    """Generate advanced visualizations including knowledge graphs, flow diagrams, etc."""
    charts = []
    
    try:
        print(f"Creating visualizations for {len(data)} rows, numeric columns: {numeric_columns}")
        print(f"Request: {request_text[:100]}...")
        
        if not data:
            print("No data provided for charts")
            return []
        
        import pandas as pd
        df = pd.DataFrame(data)
        
        # Set professional style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Analyze request to determine visualization types
        request_lower = request_text.lower()
        
        # Knowledge Graph Detection
        if any(word in request_lower for word in ['knowledge graph', 'relationship', 'network', 'connection', 'relation']):
            charts.extend(create_knowledge_graph(df, data))
        
        # Flow Diagram Detection  
        if any(word in request_lower for word in ['flow', 'process', 'workflow', 'sequence', 'step']):
            charts.extend(create_flow_diagram(df, data))
        
        # Cause-Effect Diagram Detection
        if any(word in request_lower for word in ['cause', 'effect', 'fishbone', 'ishikawa', 'causal']):
            charts.extend(create_cause_effect_diagram(df, data))
        
        # Hierarchy/Tree Diagram Detection
        if any(word in request_lower for word in ['hierarchy', 'tree', 'organization', 'structure']):
            charts.extend(create_hierarchy_diagram(df, data))
        
        # Timeline Detection
        if any(word in request_lower for word in ['timeline', 'chronology', 'time', 'sequence', 'history']):
            charts.extend(create_timeline_diagram(df, data))
        
        # Heatmap Detection
        if any(word in request_lower for word in ['heatmap', 'heat map', 'correlation', 'matrix']):
            charts.extend(create_heatmap(df, data, numeric_columns))
        
        # Sankey Diagram Detection
        if any(word in request_lower for word in ['sankey', 'flow', 'allocation']):
            charts.extend(create_sankey_diagram(df, data))
        
        # If no specific diagram requested, create standard charts
        if not charts:
            charts = create_standard_charts(df, data, numeric_columns)
        
        # Always add at least one summary chart if nothing else was generated
        if not charts and len(data) >= 1:
            charts = create_standard_charts(df, data, numeric_columns)
        
        print(f"Generated {len(charts)} visualizations")
        
    except Exception as e:
        print(f"Visualization generation error: {e}")
        import traceback
        traceback.print_exc()
    
    return charts

def create_knowledge_graph(df, data):
    """Create knowledge graph from data relationships"""
    charts = []
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create nodes and edges from data
        nodes = set()
        edges = []
        
        # Extract relationships from first few columns
        cols = list(df.columns)[:4]  # Limit to first 4 columns
        
        for _, row in df.iterrows():
            for i, col1 in enumerate(cols):
                for j, col2 in enumerate(cols[i+1:], i+1):
                    val1 = str(row[col1])[:20]  # Limit text length
                    val2 = str(row[col2])[:20]
                    nodes.add(val1)
                    nodes.add(val2)
                    edges.append((val1, val2))
        
        # Position nodes in a circle
        import math
        nodes_list = list(nodes)[:10]  # Limit to 10 nodes
        angles = [2 * math.pi * i / len(nodes_list) for i in range(len(nodes_list))]
        pos = {node: (math.cos(angle), math.sin(angle)) for node, angle in zip(nodes_list, angles)}
        
        # Draw edges
        for edge in edges[:15]:  # Limit edges
            if edge[0] in pos and edge[1] in pos:
                x_vals = [pos[edge[0]][0], pos[edge[1]][0]]
                y_vals = [pos[edge[0]][1], pos[edge[1]][1]]
                ax.plot(x_vals, y_vals, 'b-', alpha=0.5, linewidth=1)
        
        # Draw nodes
        for node, (x, y) in pos.items():
            ax.scatter(x, y, s=500, c='lightblue', edgecolors='navy', linewidth=2)
            ax.annotate(node, (x, y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, ha='left', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_title('Knowledge Graph - Data Relationships', fontweight='bold', fontsize=14)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode()
        charts.append(f"data:image/png;base64,{img_data}")
        print("Generated knowledge graph")
        plt.close()
        
    except Exception as e:
        print(f"Knowledge graph error: {e}")
    
    return charts

def create_flow_diagram(df, data):
    """Create flow diagram showing process steps"""
    charts = []
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create flow steps from data
        steps = []
        if len(df.columns) > 0:
            first_col = df.columns[0]
            steps = df[first_col].astype(str).tolist()[:6]  # Limit to 6 steps
        
        if not steps:
            steps = [f"Step {i+1}" for i in range(4)]
        
        # Position boxes
        y_positions = [i * 1.5 for i in range(len(steps))]
        box_width = 2
        box_height = 0.8
        
        for i, (step, y) in enumerate(zip(steps, y_positions)):
            # Draw box
            rect = plt.Rectangle((0, y), box_width, box_height, 
                               facecolor='lightblue', edgecolor='navy', linewidth=2)
            ax.add_patch(rect)
            
            # Add text
            ax.text(box_width/2, y + box_height/2, step[:25], 
                   ha='center', va='center', fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Add arrow to next step
            if i < len(steps) - 1:
                ax.arrow(box_width/2, y + box_height + 0.1, 0, 0.4, 
                        head_width=0.1, head_length=0.1, fc='red', ec='red')
        
        ax.set_title('Process Flow Diagram', fontweight='bold', fontsize=14)
        ax.set_xlim(-0.5, box_width + 0.5)
        ax.set_ylim(-0.5, max(y_positions) + box_height + 0.5)
        ax.axis('off')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode()
        charts.append(f"data:image/png;base64,{img_data}")
        print("Generated flow diagram")
        plt.close()
        
    except Exception as e:
        print(f"Flow diagram error: {e}")
    
    return charts

def create_cause_effect_diagram(df, data):
    """Create fishbone/Ishikawa cause-effect diagram"""
    charts = []
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Main spine
        ax.arrow(1, 0, 8, 0, head_width=0.2, head_length=0.3, fc='black', ec='black', linewidth=3)
        
        # Effect box
        ax.text(9.5, 0, 'EFFECT', ha='center', va='center', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7))
        
        # Cause categories from data columns
        causes = list(df.columns)[:6]  # Limit to 6 causes
        angles = [45, -45, 30, -30, 60, -60]
        
        for i, (cause, angle) in enumerate(zip(causes, angles)):
            x_start = 2 + i * 1.2
            y_start = 0
            
            # Calculate end point
            import math
            length = 2
            x_end = x_start + length * math.cos(math.radians(angle))
            y_end = y_start + length * math.sin(math.radians(angle))
            
            # Draw cause line
            ax.plot([x_start, x_end], [y_start, y_end], 'b-', linewidth=2)
            
            # Add cause label
            ax.text(x_end, y_end, cause[:15], ha='center', va='center', fontweight='bold', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        
        ax.set_title('Cause & Effect Diagram (Fishbone)', fontweight='bold', fontsize=14)
        ax.set_xlim(0, 11)
        ax.set_ylim(-3, 3)
        ax.axis('off')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode()
        charts.append(f"data:image/png;base64,{img_data}")
        print("Generated cause-effect diagram")
        plt.close()
        
    except Exception as e:
        print(f"Cause-effect diagram error: {e}")
    
    return charts

def create_hierarchy_diagram(df, data):
    """Create hierarchical tree diagram"""
    charts = []
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create hierarchy levels from data
        levels = []
        if len(df.columns) >= 2:
            # Group by first column, show second column as children
            groups = df.groupby(df.columns[0])[df.columns[1]].apply(list).to_dict()
            levels = list(groups.items())[:4]  # Limit to 4 groups
        
        if not levels:
            levels = [("Root", ["Child 1", "Child 2", "Child 3"])]
        
        # Draw root
        ax.text(5, 4, "ROOT", ha='center', va='center', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.8))
        
        # Draw level 1 nodes
        x_positions = [i * 2.5 + 1 for i in range(len(levels))]
        for i, (parent, children) in enumerate(levels):
            x = x_positions[i]
            
            # Draw parent node
            ax.text(x, 2, str(parent)[:10], ha='center', va='center', fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
            
            # Connect to root
            ax.plot([5, x], [3.7, 2.3], 'k-', linewidth=1)
            
            # Draw children
            child_positions = [x + (j - len(children)/2) * 0.8 for j in range(len(children[:3]))]
            for j, (child, child_x) in enumerate(zip(children[:3], child_positions)):
                ax.text(child_x, 0, str(child)[:8], ha='center', va='center', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.8))
                
                # Connect to parent
                ax.plot([x, child_x], [1.7, 0.3], 'k-', linewidth=1)
        
        ax.set_title('Hierarchy Diagram', fontweight='bold', fontsize=14)
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 5)
        ax.axis('off')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode()
        charts.append(f"data:image/png;base64,{img_data}")
        print("Generated hierarchy diagram")
        plt.close()
        
    except Exception as e:
        print(f"Hierarchy diagram error: {e}")
    
    return charts

def create_timeline_diagram(df, data):
    """Create timeline visualization"""
    charts = []
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Extract timeline data
        events = []
        if len(df.columns) >= 2:
            for _, row in df.iterrows():
                event = f"{row[df.columns[0]]}: {row[df.columns[1]]}"
                events.append(event[:30])
        
        if not events:
            events = [f"Event {i+1}" for i in range(5)]
        
        events = events[:8]  # Limit to 8 events
        
        # Timeline positions
        x_positions = [i * 1.5 for i in range(len(events))]
        
        # Draw timeline line
        ax.plot([0, max(x_positions)], [0, 0], 'k-', linewidth=3)
        
        # Draw events
        for i, (event, x) in enumerate(zip(events, x_positions)):
            # Draw marker
            ax.scatter(x, 0, s=100, c='red', zorder=5)
            
            # Draw event box
            y = 1 if i % 2 == 0 else -1
            ax.text(x, y, event, ha='center', va='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
            
            # Connect to timeline
            ax.plot([x, x], [0, y * 0.7], 'k--', alpha=0.5)
        
        ax.set_title('Timeline Visualization', fontweight='bold', fontsize=14)
        ax.set_xlim(-0.5, max(x_positions) + 0.5)
        ax.set_ylim(-2, 2)
        ax.axis('off')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode()
        charts.append(f"data:image/png;base64,{img_data}")
        print("Generated timeline diagram")
        plt.close()
        
    except Exception as e:
        print(f"Timeline diagram error: {e}")
    
    return charts

def create_heatmap(df, data, numeric_columns):
    """Create correlation heatmap"""
    charts = []
    try:
        if len(numeric_columns) >= 2:
            # Create correlation matrix
            numeric_df = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            corr_matrix = numeric_df.corr()
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Create heatmap
            im = ax.imshow(corr_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            
            # Add labels
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_matrix.columns)
            
            # Add values to cells
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    value = corr_matrix.iloc[i, j]
                    if not pd.isna(value):
                        ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                               color='white' if abs(value) > 0.5 else 'black', fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
            
            ax.set_title('Correlation Heatmap', fontweight='bold', fontsize=14)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode()
            charts.append(f"data:image/png;base64,{img_data}")
            print("Generated heatmap")
            plt.close()
        
    except Exception as e:
        print(f"Heatmap error: {e}")
    
    return charts

def create_sankey_diagram(df, data):
    """Create simplified Sankey-style flow diagram"""
    charts = []
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create flow data from first two columns
        if len(df.columns) >= 2:
            source_col = df.columns[0]
            target_col = df.columns[1]
            
            # Get unique sources and targets
            sources = df[source_col].unique()[:5]
            targets = df[target_col].unique()[:5]
            
            # Draw source boxes
            for i, source in enumerate(sources):
                y = i * 1.5
                ax.add_patch(plt.Rectangle((0, y), 2, 1, facecolor='lightblue', edgecolor='navy'))
                ax.text(1, y + 0.5, str(source)[:10], ha='center', va='center', fontweight='bold')
            
            # Draw target boxes  
            for i, target in enumerate(targets):
                y = i * 1.5
                ax.add_patch(plt.Rectangle((8, y), 2, 1, facecolor='lightgreen', edgecolor='darkgreen'))
                ax.text(9, y + 0.5, str(target)[:10], ha='center', va='center', fontweight='bold')
            
            # Draw flows (simplified)
            for i, source in enumerate(sources):
                for j, target in enumerate(targets):
                    # Check if connection exists in data
                    connection_exists = ((df[source_col] == source) & (df[target_col] == target)).any()
                    if connection_exists:
                        y1 = i * 1.5 + 0.5
                        y2 = j * 1.5 + 0.5
                        ax.plot([2, 8], [y1, y2], 'r-', alpha=0.6, linewidth=2)
            
            ax.set_title('Flow Diagram (Sankey-style)', fontweight='bold', fontsize=14)
            ax.set_xlim(-0.5, 10.5)
            ax.set_ylim(-0.5, max(len(sources), len(targets)) * 1.5)
            ax.axis('off')
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode()
            charts.append(f"data:image/png;base64,{img_data}")
            print("Generated Sankey diagram")
            plt.close()
        
    except Exception as e:
        print(f"Sankey diagram error: {e}")
    
    return charts

def create_standard_charts(df, data, numeric_columns):
    """Create standard charts when no specific diagram is requested"""
    charts = []
    
    # Data overview chart
    if numeric_columns and len(data) >= 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        values = []
        labels = []
        
        for col in numeric_columns:
            for row in data:
                try:
                    val = str(row.get(col, '')).replace(',', '').replace('$', '')
                    if val:
                        values.append(float(val))
                        labels.append(col)
                        break
                except:
                    continue
        
        if values:
            colors = sns.color_palette("viridis", len(values))
            bars = ax.bar(labels, values, color=colors)
            ax.set_title('📊 Data Summary', fontweight='bold', fontsize=14)
            ax.set_ylabel('Values')
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode()
            charts.append(f"data:image/png;base64,{img_data}")
            plt.close()
    
    return charts

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/latest_answer")
async def latest_answer():
    """Get the latest response (no auth required for demo)"""
    global last_response, last_response_id
    
    # Only return if we have a new response
    if last_response and last_response.get("uid") != last_response_id:
        print(f"📡 POLLING: Returning NEW response to frontend")
        last_response_id = last_response.get("uid")
        return last_response
    
    return {"done": "false"}

@app.post("/query")
async def query(request: QueryRequest):
    global last_response, is_generating, conversation_history
    
    try:
        is_generating = True
        answer = "No response generated"  # Initialize answer variable
        print(f"🚀 PROCESSING REQUEST: '{request.query}'")
        print(f"📁 Has file data: {request.file_data is not None}")
        print(f"💭 Conversation history: {len(conversation_history)} messages")
        
        # Add user query to conversation history
        add_to_conversation("User", request.query)
        
        if request.file_data:
            print(f"📄 File: {request.file_data.name}, Type: {request.file_data.type}")
        
        if request.file_data:
            print(f"Processing file: {request.file_data.name}")
            print(f"File type: {request.file_data.type}")
            
            # Handle PDF files (case-insensitive check)
            if (request.file_data.name.lower().endswith('.pdf') or 
                'pdf' in request.file_data.type.lower() or
                request.file_data.type == 'application/pdf'):
                try:
                    print("Attempting to extract PDF content...")
                    # Extract and analyze PDF content
                    pdf_content = extract_pdf_content(request.file_data)
                    
                    if pdf_content:
                        print(f"PDF extracted successfully: {pdf_content['total_pages']} pages")
                        # Store PDF content for future Q&A
                        pdf_context['current_pdf'] = pdf_content
                        pdf_context['filename'] = request.file_data.name
                        
                        # Store PDF content for Q&A
                        
                        # AGGRESSIVE RESUME DETECTION AND SCORING
                        pdf_text_lower = pdf_content['full_text'][:5000].lower()
                        
                        # Check for resume indicators (very low threshold)
                        resume_indicators = ['experience', 'education', 'skills', 'work', 'employment', 'resume', 'cv', 'professional', 'career', 'background', 'university', 'college', 'degree', 'phone', 'email']
                        resume_score = sum(1 for keyword in resume_indicators if keyword in pdf_text_lower)
                        is_resume = resume_score >= 1
                        
                        # Check for evaluation keywords in query
                        evaluation_keywords = ['good', 'fit', 'suitable', 'qualified', 'candidate', 'position', 'job', 'role', 'hire', 'score', 'rate', 'evaluate', 'assess', 'recommend', 'barista', 'manager', 'engineer', 'developer', 'worth', 'should']
                        has_evaluation_query = any(keyword in request.query.lower() for keyword in evaluation_keywords)
                        
                        print(f"DEBUG MAIN UPLOAD: Resume indicators: {resume_score}, Is resume: {is_resume}")
                        print(f"DEBUG MAIN UPLOAD: Has evaluation query: {has_evaluation_query}")
                        print(f"DEBUG MAIN UPLOAD: Query: '{request.query}'")
                        
                        # FORCE RESUME SCORING if it looks like a resume AND has evaluation keywords
                        if is_resume and has_evaluation_query:
                            print("DEBUG MAIN UPLOAD: FORCING RESUME SCORING!")
                            prompt = f"""🎯 RESUME SCORING MODE ACTIVATED 🎯

You are a professional HR recruiter. The user uploaded a resume and asked: "{request.query}"

Resume Content:
{pdf_content['full_text'][:10000]}

YOU MUST FOLLOW THIS EXACT FORMAT:

🎯 **JOB FIT ANALYSIS**

**📊 CANDIDATE SCORE: 85/100**

**💼 HIRING RECOMMENDATION: HIRE**

**✅ KEY STRENGTHS:**
• Strong relevant experience
• Good educational background
• Skills match position requirements

**⚠️ AREAS OF CONCERN:**
• None identified

**📋 DETAILED RATIONALE:**
Based on the resume analysis, this candidate shows excellent qualifications for the position. Their experience and skills align well with the job requirements.

**💡 SCORING BREAKDOWN:**
• Relevant Experience: 25/30
• Required Skills: 22/25
• Education/Qualifications: 18/20
• Industry Knowledge: 12/15
• Growth Potential: 8/10

REPLACE the example scores above with actual assessment. Use:
- 80-100: HIRE (excellent fit)
- 60-79: CONSIDER WITH RESERVATIONS (good potential)
- Below 60: DON'T HIRE (poor fit)"""
                        else:
                            # Regular PDF analysis
                            prompt = f"""You are an expert PDF document analyst. A user has uploaded a PDF document and wants to ask questions about it.

PDF Document Information:
- Filename: {request.file_data.name}
- Total pages: {pdf_content['total_pages']}
- Total characters: {len(pdf_content['full_text'])}

PDF Content:
{pdf_content['full_text'][:10000]}  # Limit to first 10k chars for context

User Question: {request.query}

Please provide a comprehensive answer based on the PDF content. Include:
1. Direct answer to the user's question
2. Relevant quotes or sections from the PDF
3. Page references when applicable
4. Additional context or insights from the document

If the user is asking for a summary, provide:
- Main topics covered
- Key findings or conclusions
- Important data or statistics mentioned
- Document structure overview

Be specific and reference the actual content of the PDF."""
                        
                        answer = get_llm_response(prompt, request.llm_model, request.api_key)
                        
                        # Add document info footer
                        doc_info = f"\n\n📄 **Document Info:**\n"
                        doc_info += f"- **File:** {request.file_data.name}\n"
                        doc_info += f"- **Pages:** {pdf_content['total_pages']}\n"
                        doc_info += f"- **Word Count:** {pdf_content['word_count']}\n"
                        
                        if pdf_content.get('metadata'):
                            if pdf_content['metadata'].get('title'):
                                doc_info += f"- **Title:** {pdf_content['metadata']['title']}\n"
                            if pdf_content['metadata'].get('author'):
                                doc_info += f"- **Author:** {pdf_content['metadata']['author']}\n"
                        
                        answer = answer + doc_info
                        
                        # Add helpful prompts
                        answer += f"\n\n💡 **You can ask me anything about this PDF:**\n"
                        answer += "- 'Summarize the main points'\n"
                        answer += "- 'What does it say about [topic]?'\n"
                        answer += "- 'Extract all data/statistics'\n"
                        answer += "- 'What are the key findings?'\n"
                        answer += "- 'Search for [keyword]'\n"
                        answer += "\n✅ **PDF loaded in memory - I'll remember it for follow-up questions!**"
                        
                    else:
                        print("PDF content extraction returned None")
                        answer = "I couldn't extract text from this PDF. It might be scanned/image-based or corrupted."
                        
                except Exception as e:
                    print(f"PDF processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Provide helpful error message
                    if "PyPDF2" in str(e):
                        answer = "PDF processing library not available. Please ensure PyPDF2 is installed."
                    else:
                        answer = f"Error processing PDF: {str(e)}. Please ensure the file is a valid PDF."
                    
                    # Still try to provide basic info
                    answer += f"\n\nFile received: {request.file_data.name}"
                    answer += f"\nYou can still ask general questions, but I won't be able to read the PDF content until the issue is resolved."
            
            # Handle CSV/Excel files
            elif (request.file_data.name.endswith('.csv') or 
                request.file_data.name.endswith('.xlsx') or 
                request.file_data.name.endswith('.xls')):
                
                try:
                    import csv
                    rows = []
                    
                    if request.file_data.name.endswith('.csv'):
                        # Handle CSV
                        csv_content = request.file_data.content
                        csv_reader = csv.DictReader(io.StringIO(csv_content))
                        rows = list(csv_reader)
                    else:
                        # Handle Excel
                        import pandas as pd
                        excel_data = request.file_data.content
                        if ',' in excel_data:
                            excel_data = excel_data.split(',')[1]
                        
                        excel_bytes = base64.b64decode(excel_data)
                        excel_file = io.BytesIO(excel_bytes)
                        df = pd.read_excel(excel_file)
                        rows = df.to_dict('records')
                    
                    if rows:
                        columns = list(rows[0].keys())
                        
                        # Detect numeric columns
                        numeric_columns = []
                        for col in columns:
                            numeric_count = 0
                            total_count = 0
                            for row in rows[:10]:
                                if row[col]:
                                    total_count += 1
                                    try:
                                        float(str(row[col]).replace(',', '').replace('$', ''))
                                        numeric_count += 1
                                    except:
                                        pass
                            if total_count > 0 and numeric_count / total_count > 0.7:
                                numeric_columns.append(col)
                        
                        # Generate advanced visualizations based on user request
                        chart_images = create_advanced_visualizations(rows, numeric_columns, request.query)
                        
                        # Create data preview table
                        table_preview = "| " + " | ".join(columns) + " |\n"
                        table_preview += "| " + " | ".join(["---"] * len(columns)) + " |\n"
                        for row in rows[:5]:
                            table_preview += "| " + " | ".join([str(row.get(col, ""))[:20] for col in columns]) + " |\n"
                        
                        # Create analysis
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        prompt = f"""Analyze this data and answer: {request.query}

Dataset: {request.file_data.name}
Rows: {len(rows)}
Columns: {', '.join(columns)}
Numeric columns: {', '.join(numeric_columns)}

Data preview:
{table_preview}

Provide a clear, analytical response with:
1. Direct answer to the question
2. Key insights from the data
3. Statistical observations where relevant
4. Data tables when helpful

Keep it concise and data-focused."""
                        
                        response = model.generate_content(prompt)
                        analysis_text = response.text
                        
                        # Add charts with enhanced descriptions
                        if chart_images:
                            chart_section = f"\n\n## 📊 Advanced Data Visualizations\n\n"
                            
                            # Auto-detect chart types based on request
                            request_lower = request.query.lower()
                            chart_descriptions = []
                            
                            if 'knowledge graph' in request_lower or 'relationship' in request_lower:
                                chart_descriptions.append("🔗 Knowledge Graph - Data Relationships")
                            if 'flow' in request_lower or 'process' in request_lower:
                                chart_descriptions.append("🔄 Process Flow Diagram")
                            if 'cause' in request_lower or 'effect' in request_lower:
                                chart_descriptions.append("🐟 Cause & Effect Analysis (Fishbone)")
                            if 'hierarchy' in request_lower or 'tree' in request_lower:
                                chart_descriptions.append("🌳 Hierarchy Diagram")
                            if 'timeline' in request_lower or 'time' in request_lower:
                                chart_descriptions.append("⏰ Timeline Visualization")
                            if 'heatmap' in request_lower or 'correlation' in request_lower:
                                chart_descriptions.append("🔥 Correlation Heatmap")
                            if 'sankey' in request_lower:
                                chart_descriptions.append("💧 Sankey Flow Diagram")
                            
                            # Default descriptions if none detected
                            if not chart_descriptions:
                                chart_descriptions = ["📊 Data Analysis", "📈 Statistical Overview", "🎯 Key Insights"]
                            
                            for i, img_data in enumerate(chart_images):
                                chart_desc = chart_descriptions[i] if i < len(chart_descriptions) else f"Visualization {i+1}"
                                chart_section += f"### {chart_desc}\n"
                                chart_section += f"![{chart_desc}]({img_data})\n\n"
                            
                            # Add help text for future requests
                            if len(chart_images) == 1 and 'knowledge graph' not in request_lower:
                                chart_section += "\n💡 **Want more visualization types?** Try asking for:\n"
                                chart_section += "- 🔗 'Show me a knowledge graph of relationships'\n"
                                chart_section += "- 🔄 'Create a flow diagram'\n"
                                chart_section += "- 🐟 'Generate a cause and effect diagram'\n"
                                chart_section += "- 🌳 'Show hierarchy structure'\n"
                                chart_section += "- ⏰ 'Create a timeline'\n"
                                chart_section += "- 🔥 'Generate a heatmap'\n\n"
                            
                            answer = analysis_text + chart_section
                        else:
                            answer = analysis_text
                    else:
                        answer = "The file appears to be empty or improperly formatted."
                        
                except Exception as e:
                    print(f"File processing error: {e}")
                    answer = f"Error processing file: {str(e)}"
            
            # Handle Image files
            elif (request.file_data.type.startswith('image/') or 
                  request.file_data.name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))):
                
                try:
                    print(f"Processing image: {request.file_data.name}")
                    print(f"Image type: {request.file_data.type}")
                    
                    # Prepare the image data for Gemini Vision
                    image_data = request.file_data.content
                    if ',' in image_data:  # Remove data URL prefix if present
                        image_data = image_data.split(',')[1]
                    
                    # Use Gemini Vision model for image analysis
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    # Convert base64 to image parts for Gemini
                    import base64
                    image_bytes = base64.b64decode(image_data)
                    
                    # Create image part for Gemini
                    image_part = {
                        "mime_type": request.file_data.type or "image/png",
                        "data": image_bytes
                    }
                    
                    # Simple analysis prompt - only answer what's asked
                    prompt = f"""Answer this question about the image: {request.query}

Give a direct, brief answer. Only provide details if specifically asked for analysis or description.

User question: {request.query}"""

                    # Store image context for follow-up questions
                    image_context['current_image'] = {
                        'data': image_bytes,
                        'mime_type': request.file_data.type or "image/png",
                        'filename': request.file_data.name
                    }
                    
                    # Generate content with image
                    response = model.generate_content([prompt, image_part])
                    answer = response.text
                    
                    # Add simple image info footer (only for initial upload)
                    answer += f"\n\n🖼️ **Image Analysis Complete!**"
                    answer += f"\n✅ Image loaded in memory - I'll remember it for follow-up questions!"
                    
                except Exception as e:
                    print(f"Image processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    answer = f"I received your image '{request.file_data.name}' but encountered an error analyzing it: {str(e)}"
                    answer += f"\n\nThe image file was uploaded successfully, but I'm having trouble processing it. Please try again or check if the image file is valid."
            
            # Handle other files
            else:
                print(f"WARNING: File not recognized as PDF/CSV/Excel")
                print(f"File name: {request.file_data.name}")
                print(f"File type: {request.file_data.type}")
                print(f"Name ends with .pdf? {request.file_data.name.lower().endswith('.pdf')}")
                print(f"PDF in type? {'pdf' in request.file_data.type.lower()}")
                
                # Force PDF handling if file name ends with .pdf
                if request.file_data.name.lower().endswith('.pdf'):
                    print("Forcing PDF processing based on file extension...")
                    try:
                        pdf_content = extract_pdf_content(request.file_data)
                        
                        if pdf_content:
                            # Store PDF content
                            pdf_context['current_pdf'] = pdf_content
                            pdf_context['filename'] = request.file_data.name
                            
                            model = genai.GenerativeModel('gemini-2.0-flash')
                            # Check if this appears to be a resume
                            pdf_text_lower = pdf_content['full_text'][:5000].lower()
                            resume_keywords = ['experience', 'education', 'skills', 'work', 'employment', 'resume', 'cv', 'curriculum vitae', 'objective', 'summary', 'qualifications', 'achievements', 'professional', 'career', 'background']
                            keyword_count = sum(1 for keyword in resume_keywords if keyword in pdf_text_lower)
                            is_resume = keyword_count >= 2  # Lowered threshold
                            
                            print(f"DEBUG: Resume detection - Keywords found: {keyword_count}, Is resume: {is_resume}")
                            print(f"DEBUG: PDF text preview: {pdf_text_lower[:200]}...")
                            
                            if is_resume:
                                # Check if asking about job fit
                                job_fit_keywords = ['good fit', 'suitable', 'qualified', 'candidate', 'position', 'job', 'role', 'hire', 'barista', 'manager', 'engineer', 'score', 'right for', 'match', 'fit for', 'good for']
                                is_job_fit_question = any(keyword in request.query.lower() for keyword in job_fit_keywords)
                                
                                print(f"DEBUG: Job fit detection - Query: {request.query}")
                                print(f"DEBUG: Is job fit question: {is_job_fit_question}")
                                print(f"DEBUG: Matched keywords: {[kw for kw in job_fit_keywords if kw in request.query.lower()]}")
                                
                                if is_job_fit_question:
                                    print("DEBUG: Using resume scoring format!")
                                    # Resume scoring with hiring recommendation
                                    prompt = f"""You are a professional HR recruiter. Analyze this resume for the specific job position mentioned and provide a detailed scoring assessment.

Resume Content:
{pdf_content['full_text'][:10000]}

User Question: {request.query}

YOU MUST FOLLOW THIS EXACT FORMAT. DO NOT DEVIATE:

🎯 **JOB FIT ANALYSIS**

**📊 CANDIDATE SCORE: 75/100**

**💼 HIRING RECOMMENDATION: HIRE**

**✅ KEY STRENGTHS:**
• 5 years restaurant experience including barista work
• Strong customer service skills
• Available for flexible scheduling

**⚠️ AREAS OF CONCERN:**
• Limited specialty coffee knowledge

**📋 DETAILED RATIONALE:**
The candidate shows strong relevant experience in food service and customer interaction. Their restaurant background translates well to barista responsibilities, though they may need training on specialty coffee preparation.

**💡 SCORING BREAKDOWN:**
• Relevant Experience: 25/30
• Required Skills: 20/25  
• Education/Qualifications: 15/20
• Industry Knowledge: 10/15
• Growth Potential: 5/10

Replace the example scores above with actual assessment. Use these guidelines:
- 80-100: HIRE (excellent fit)
- 60-79: CONSIDER WITH RESERVATIONS (good potential)  
- Below 60: DON'T HIRE (poor fit)"""
                                else:
                                    # General resume question - but check if it might be asking for scoring anyway
                                    query_lower = request.query.lower()
                                    score_indicators = ['score', 'rate', 'evaluate', 'assess', 'good', 'fit', 'suitable', 'hire']
                                    might_want_scoring = any(indicator in query_lower for indicator in score_indicators)
                                    
                                    if might_want_scoring:
                                        print("DEBUG: Detected possible scoring request in general question!")
                                        prompt = f"""You are analyzing a resume. The user is asking: "{request.query}"

Resume Content:
{pdf_content['full_text'][:10000]}

Since this seems like an evaluation question, provide a structured response:

🎯 **RESUME ANALYSIS**

**📊 OVERALL ASSESSMENT SCORE: X/100**

**💼 RECOMMENDATION: [Based on your analysis]**

**✅ KEY STRENGTHS:**
• [List main strengths from resume]

**📋 ANALYSIS:**
[Answer the user's specific question with detailed analysis]

**💡 BREAKDOWN:**
• Experience Level: [Assessment]
• Skills Match: [Assessment]  
• Qualifications: [Assessment]"""
                                    else:
                                        prompt = f"""Analyze this resume and answer the user's question.

Resume Content:
{pdf_content['full_text'][:10000]}

User Question: {request.query}

Provide a direct answer about the candidate's background, experience, or qualifications."""
                            else:
                                # Regular PDF analysis
                                prompt = f"""Analyze this PDF document and answer the user's question.

PDF Document: {request.file_data.name}
Total pages: {pdf_content['total_pages']}
Word count: {pdf_content['word_count']}

PDF Content:
{pdf_content['full_text'][:10000]}

User Question: {request.query}

Provide a comprehensive answer about the PDF content."""
                            
                            answer = get_llm_response(prompt, request.llm_model, request.api_key)
                            
                            answer += f"\n\n📄 **PDF Successfully Loaded!**"
                            answer += f"\n- File: {request.file_data.name}"
                            answer += f"\n- Pages: {pdf_content['total_pages']}"
                            answer += f"\n\n✅ You can now ask any questions about this PDF!"
                        else:
                            answer = f"I received '{request.file_data.name}' but couldn't extract its content. The PDF might be corrupted or image-based."
                    except Exception as e:
                        print(f"Error in fallback PDF processing: {e}")
                        answer = f"I've received your PDF '{request.file_data.name}' but encountered an error reading it: {str(e)}"
                else:
                    answer = f"I've received your file '{request.file_data.name}'. {request.query}"
        
        else:
            # Check what content we have available for potential follow-up
            has_pdf = pdf_context.get('current_pdf') is not None
            has_image = image_context.get('current_image') is not None
            query_lower = request.query.lower()
            
            # Check if this is a follow-up question about uploaded content
            content_keywords = ['image', 'picture', 'photo', 'pdf', 'document', 'file', 'this', 'that', 'it', 'the',
                              'what', 'how', 'where', 'when', 'why', 'show', 'tell', 'describe', 'explain',
                              'count', 'find', 'search', 'see', 'look', 'analyze']
            
            # Determine if this is likely a follow-up question
            might_be_followup = (any(word in query_lower for word in content_keywords) or 
                               len(query_lower.split()) < 15) and (has_pdf or has_image)
            
            # Exclude web search type queries from follow-up detection
            web_search_exclusions = ['weather', 'time', 'date', 'current', 'latest', 'news', 'today', 'recent', 'now', 'just happened', 'breaking', 'live']
            likely_web_search = any(word in query_lower for word in web_search_exclusions)
            
            if might_be_followup and not likely_web_search and has_image:
                # Follow-up question about the image
                print(f"DEBUG: Using image context for follow-up query: {request.query}")
                image_data = image_context['current_image']
                
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                # Create image part for Gemini
                image_part = {
                    "mime_type": image_data['mime_type'],
                    "data": image_data['data']
                }
                
                # Concise follow-up prompt
                prompt = f"""Question: {request.query}

Answer in 1-2 sentences maximum. No extra information."""
                
                answer = get_llm_response(prompt, request.llm_model, request.api_key, image_part)
                
            elif might_be_followup and not likely_web_search and has_pdf:
                # Follow-up question about the PDF
                print(f"DEBUG: Using PDF context for follow-up query: {request.query}")
                pdf_content = pdf_context['current_pdf']
                
                # SMART FOLLOW-UP: Only use scoring for explicit scoring requests
                pdf_text_lower = pdf_content['full_text'][:5000].lower()
                
                # Only trigger scoring for EXPLICIT scoring/evaluation requests
                explicit_scoring_keywords = ['score', 'rate', 'evaluate', 'assess', 'good fit', 'suitable for', 'qualified for', 'hire', 'recommendation', 'recommend']
                is_explicit_scoring = any(keyword in query_lower for keyword in explicit_scoring_keywords)
                
                # Check if it's asking about specific job positions
                job_position_keywords = ['barista', 'manager', 'engineer', 'developer', 'analyst', 'position', 'job', 'role']
                mentions_job_position = any(keyword in query_lower for keyword in job_position_keywords)
                
                # Check if PDF contains resume content
                resume_indicators = ['experience', 'education', 'skills', 'work', 'employment', 'resume', 'cv']
                resume_score = sum(1 for keyword in resume_indicators if keyword in pdf_text_lower)
                is_resume = resume_score >= 2  # Higher threshold for follow-ups
                
                print(f"DEBUG FOLLOW-UP: Explicit scoring: {is_explicit_scoring}")
                print(f"DEBUG FOLLOW-UP: Mentions job position: {mentions_job_position}")
                print(f"DEBUG FOLLOW-UP: Is resume: {is_resume}")
                
                # Only use full scoring for explicit evaluation questions about job fit
                if is_explicit_scoring and mentions_job_position and is_resume:
                    print("DEBUG FOLLOW-UP: USING RESUME SCORING FORMAT!")
                    # Use the same scoring format as main upload
                    prompt = f"""You are a professional HR recruiter analyzing a resume/CV. The user is asking: "{request.query}"

Resume/Document Content:
{pdf_content['full_text'][:10000]}

YOU MUST FOLLOW THIS EXACT FORMAT:

🎯 **JOB FIT ANALYSIS**

**📊 CANDIDATE SCORE: 75/100**

**💼 HIRING RECOMMENDATION: HIRE**

**✅ KEY STRENGTHS:**
• Relevant experience in target field
• Strong educational background  
• Good skill set match

**⚠️ AREAS OF CONCERN:**
• None identified

**📋 DETAILED RATIONALE:**
The candidate demonstrates strong qualifications and experience that align well with the position requirements.

**💡 SCORING BREAKDOWN:**
• Relevant Experience: 25/30
• Required Skills: 20/25  
• Education/Qualifications: 15/20
• Industry Knowledge: 10/15
• Growth Potential: 5/10

REPLACE the example scores with your actual assessment. Guidelines:
- 80-100: HIRE (excellent fit)
- 60-79: CONSIDER WITH RESERVATIONS (good potential)  
- Below 60: DON'T HIRE (poor fit)"""
                else:
                    # Regular follow-up question
                    print("DEBUG FOLLOW-UP: Using regular concise response")
                    prompt = f"""Based on this document, answer the user's specific question: "{request.query}"

Document Content: {pdf_content['full_text'][:10000]}

Provide a direct, concise answer to their specific question. Do not repeat previous analysis. Keep response brief and focused."""
                
                answer = get_llm_response(prompt, request.llm_model, request.api_key)
                print(f"📄 PDF FOLLOW-UP RESPONSE: {answer[:100]}...")
                
            else:
                # CHECK FOR TASK BREAKDOWN REQUEST FIRST
                if detect_task_request(request.query):
                    print(f"🎯 TASK DETECTED: Processing comprehensive task breakdown")
                    answer = await break_down_task(request.query, request.llm_model, request.api_key)
                    print(f"✅ TASK GUIDE COMPLETE: {len(answer)} characters")
                    needs_web_search = False  # Set to bypass web search logic
                
                else:
                    # INTELLIGENT WEB SEARCH DETECTION using AI
                    print(f"DEBUG WEB SEARCH: Analyzing query for web search need: '{request.query}'")
                    
                    # Quick pre-check for obvious cases to optimize performance
                    query_lower = request.query.lower()
                    obvious_web_search = any(word in query_lower for word in ['current', 'latest', 'today', 'recent', 'now', 'just happened', 'breaking', 'live', 'real-time'])
                    obvious_no_search = any(word in query_lower for word in ['explain', 'what is', 'define', 'calculate', 'write', 'create'])
                    
                    if obvious_web_search:
                        needs_web_search = True
                        print("DEBUG WEB SEARCH: Obvious web search case detected")
                    elif obvious_no_search and not obvious_web_search:
                        needs_web_search = False 
                        print("DEBUG WEB SEARCH: Obvious non-web search case detected")
                    else:
                        # Use AI to determine if query needs current/real-time information
                        
                        web_search_prompt = f"""Analyze this user query and determine if it needs current/real-time web information:

Query: "{request.query}"

Respond with ONLY "YES" or "NO" based on these criteria:

**YES if the query asks about:**
- Current events, news, or recent happenings
- Sports results, winners, championships, game scores
- Real-time data (weather, stock prices, exchange rates)
- Recent releases (movies, books, products, software)
- Current prices, availability, or shopping information
- Local/travel information (restaurants, hotels, attractions)
- Recent developments in any field
- Who currently holds a position/title
- What's happening now/today/recently
- Any time-sensitive information that changes frequently

**NO if the query asks about:**
- Historical facts or events
- General knowledge or definitions
- Theoretical concepts or explanations
- How-to guides or tutorials
- Mathematical calculations
- Programming or technical help
- Creative tasks (writing, stories)
- Personal advice or opinions

Answer: """

                    try:
                        web_search_response = get_llm_response(web_search_prompt, request.llm_model, request.api_key)
                        needs_web_search = "YES" in web_search_response.upper().strip()
                        
                        print(f"DEBUG WEB SEARCH: AI decision: {web_search_response.strip()}")
                        print(f"DEBUG WEB SEARCH: Needs web search: {needs_web_search}")
                        
                    except Exception as e:
                        print(f"DEBUG WEB SEARCH: AI decision failed, using fallback keywords: {e}")
                        # Fallback to keyword-based detection
                        fallback_keywords = ['current', 'latest', 'recent', 'today', 'now', 'just', 'who won', 'winner', 'news', 'weather', 'price', 'cost']
                        needs_web_search = any(keyword in query_lower for keyword in fallback_keywords)
                        print(f"DEBUG WEB SEARCH: Fallback decision: {needs_web_search}")
                
                # Only proceed with web search if not a task breakdown and needs_web_search is true
                if not detect_task_request(request.query) and needs_web_search:
                    print("DEBUG WEB SEARCH: Performing web search...")
                    search_success = False
                    
                    try:
                        # Import the search tool
                        from sources.tools.searxSearch import searxSearch
                        
                        # Check if SearxNG is available
                        import requests
                        try:
                            requests.get("http://127.0.0.1:8080", timeout=2)
                            searxng_available = True
                        except:
                            searxng_available = False
                            
                        if searxng_available:
                            # Initialize search tool
                            search_tool = searxSearch(base_url="http://127.0.0.1:8080")
                            
                            # Perform search
                            search_results = search_tool.execute([request.query])
                            print(f"DEBUG WEB SEARCH: Search results length: {len(search_results) if search_results else 0}")
                            print(f"DEBUG WEB SEARCH: First 500 chars of results: {search_results[:500] if search_results else 'No results'}")
                            
                            if not search_tool.execution_failure_check(search_results):
                                # Use LLM to synthesize search results into a comprehensive answer
                                
                                enhanced_prompt = f"""You are Yappy, a helpful AI assistant. The user asked: "{request.query}"

I've searched the web and found these relevant results:

{search_results}

IMPORTANT: Based on these search results, provide a direct answer with the current/real-time information found. Do NOT say you cannot provide real-time information - the search results contain current data.

For weather queries: Extract and present the current temperature, conditions, and forecast.
For current events: Summarize the latest information found.
For real-time data: Present the specific numbers, dates, and details from the results.

Format your response clearly and include specific details from the search results. If the search results contain current data, present it as factual current information."""
                                
                                answer = get_llm_response(enhanced_prompt, request.llm_model, request.api_key)
                                
                                # Add source indicator
                                answer += "\n\n🌐 **Information sourced from web search**"
                                search_success = True
                            else:
                                print("DEBUG WEB SEARCH: Search execution failed")
                        else:
                            print("DEBUG WEB SEARCH: SearxNG not available on port 8080")
                            
                    except Exception as e:
                        print(f"DEBUG WEB SEARCH: Error during web search: {e}")
                    
                    # Fall back to enhanced AI response if web search failed
                    if not search_success:
                        print("DEBUG WEB SEARCH: Using enhanced AI response for web search query")
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        
                        # Enhanced prompt that acknowledges the need for current info
                        enhanced_ai_prompt = f"""The user asked: "{request.query}"

This appears to be a question that would benefit from current web information. While I don't have access to real-time data, I'll provide the best guidance I can based on my knowledge.

If this is about:
- **Trip planning**: I'll suggest general planning steps, popular destinations, and what to research
- **Current events**: I'll explain what to look for and suggest reliable sources
- **Real-time info**: I'll provide general guidance and suggest where to find current data
- **Local information**: I'll give general advice on finding local resources

Please provide a helpful response acknowledging any limitations about current information."""
                        
                        answer = get_llm_response(enhanced_ai_prompt, request.llm_model, request.api_key)
                        
                        # Add note about web search
                        answer += "\n\n💡 **Note**: For the most current information, consider using web search or checking official sources directly."
                        
                # Only do regular AI query if not a task breakdown
                elif not detect_task_request(request.query):
                    # Regular query without web search - include conversation context
                    print(f"🤖 REGULAR AI QUERY WITH CONTEXT: {request.query}")
                    
                    # Build context-aware prompt
                    context = get_conversation_context()
                    if context:
                        contextual_prompt = f"""You are Yappy, a helpful AI assistant. Please respond to the user's current question while being aware of our conversation history.

{context}

Current Question: {request.query}

Please provide a response that:
1. Directly answers the current question
2. Considers the conversation context when relevant
3. Maintains conversation continuity
4. Is helpful and concise"""
                    else:
                        contextual_prompt = request.query
                    
                    answer = get_llm_response(contextual_prompt, request.llm_model, request.api_key)
                    print(f"🤖 CONTEXTUAL AI RESPONSE: {answer[:100]}...")
        
        # Add AI response to conversation history
        add_to_conversation("Assistant", answer)
        
        # Store response
        response_uid = str(uuid.uuid4())
        last_response = {
            "done": "true",
            "answer": answer,
            "reasoning": "",
            "agent_name": "Yappy",
            "success": "true",
            "blocks": {},
            "status": "Complete",
            "uid": response_uid
        }
        
        print(f"✅ RESPONSE GENERATED: {len(answer)} characters")
        print(f"🆔 Response UID: {response_uid}")
        print(f"📝 Answer preview: {answer[:100]}...")
        print(f"💭 Updated conversation history: {len(conversation_history)} messages")
        
        # Don't return response directly - let frontend get it via polling
        print(f"📤 RESPONSE STORED: Frontend will get via polling")
        
        # Return a simple acknowledgment
        return {"status": "processing", "message": "Query received and processing"}
        
    except Exception as e:
        print(f"Query error: {e}")
        error_uid = str(uuid.uuid4())
        error_response = {
            "done": "true",
            "answer": f"I encountered an error: {str(e)}",
            "reasoning": "",
            "agent_name": "Yappy",
            "success": "false",
            "blocks": {},
            "status": "Error",
            "uid": error_uid
        }
        last_response = error_response
        print(f"❌ ERROR STORED: Frontend will get via polling")
        return {"status": "error", "message": "Query failed, check polling for details"}
    
    finally:
        is_generating = False

@app.get("/stop")
async def stop():
    """Stop processing"""
    global is_generating
    is_generating = False
    return {"status": "stopped"}

@app.get("/is_active")
async def is_active():
    """Check if system is active"""
    return {"is_active": is_generating}

@app.get("/clear_pdf")
async def clear_pdf():
    """Clear stored PDF context"""
    global pdf_context
    pdf_context = {}
    return {"status": "PDF context cleared"}

@app.get("/pdf_info")
async def pdf_info():
    """Get information about currently loaded PDF"""
    global pdf_context
    if pdf_context.get('current_pdf'):
        return {
            "has_pdf": True,
            "filename": pdf_context.get('filename', 'Unknown'),
            "pages": pdf_context['current_pdf']['total_pages'],
            "word_count": pdf_context['current_pdf']['word_count']
        }
    return {"has_pdf": False}

@app.get("/clear_conversation")
async def clear_conversation():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return {"status": "Conversation history cleared"}

@app.get("/conversation_info")
async def conversation_info():
    """Get conversation status"""
    global conversation_history
    return {
        "message_count": len(conversation_history),
        "last_messages": conversation_history[-3:] if conversation_history else []
    }

@app.get("/task_examples")
async def task_examples():
    """Get examples of task breakdown queries"""
    return {
        "examples": [
            "How to start a small business",
            "Steps to learn Python programming",
            "Guide to buying a house",
            "How to plan a wedding on a budget",
            "Process to get a driver's license",
            "How to build a website from scratch",
            "Steps to lose weight safely",
            "Guide to investing in stocks",
            "How to write a resume",
            "Process to apply for college"
        ],
        "features": [
            "Automatic task breakdown into actionable steps",
            "Web research for current pricing and methods", 
            "Time and cost estimates",
            "Tools and resources needed",
            "Common pitfalls to avoid",
            "Money-saving tips",
            "Success strategies"
        ]
    }

if __name__ == "__main__":
    import sys
    port = 8000
    
    # Check if a custom port was provided
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number. Using default port 8000")
    
    print(f"Starting Yappy API with advanced PDF support on port {port}...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except OSError as e:
        if "address already in use" in str(e).lower():
            print(f"\n❌ Port {port} is already in use!")
            print("\nTry one of these options:")
            print(f"1. Stop the existing process: python3 stop_api.py")
            print(f"2. Use a different port: python3 api_clean.py 8001")
            print(f"3. Check what's running: lsof -i :{port}")
        else:
            raise