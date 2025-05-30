#!/usr/bin/env python3
"""
Fixed API for Neo with working image upload
"""

import os
import uuid
import json
import base64
import io
import tempfile
import warnings
from typing import Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai

# For visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

warnings.filterwarnings("ignore")

def generate_charts_base64(data, numeric_columns, file_name="chart"):
    """Generate charts as base64 encoded images for embedding in HTML"""
    charts = []
    
    try:
        if not data or len(data) < 2:
            return []
        
        # Convert data to DataFrame for easier plotting
        import pandas as pd
        df = pd.DataFrame(data)
        
        # Set style for better-looking charts
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Histogram for numeric columns (if any)
        if len(numeric_columns) > 0:
            fig, axes = plt.subplots(1, min(len(numeric_columns), 2), figsize=(12, 5))
            if len(numeric_columns) == 1:
                axes = [axes]
            
            for i, col in enumerate(numeric_columns[:2]):  # Max 2 columns for space
                values = []
                for row in data:
                    try:
                        val = str(row.get(col, '')).replace(',', '').replace('$', '')
                        if val:
                            values.append(float(val))
                    except:
                        pass
                
                if values and len(values) > 1:
                    ax = axes[i] if len(numeric_columns) > 1 else axes[0]
                    ax.hist(values, bins=min(20, len(set(values))), alpha=0.7, color=sns.color_palette()[i])
                    ax.set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            charts.append(f"data:image/png;base64,{img_base64}")
            plt.close()
        
        # 2. Bar chart for categorical data
        categorical_cols = [col for col in df.columns if col not in numeric_columns]
        if categorical_cols:
            plt.figure(figsize=(10, 6))
            col = categorical_cols[0]
            value_counts = df[col].value_counts().head(10)
            
            colors = sns.color_palette("viridis", len(value_counts))
            bars = plt.bar(range(len(value_counts)), value_counts.values, color=colors)
            plt.title(f'{col} Distribution (Top 10)', fontsize=14, fontweight='bold')
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            charts.append(f"data:image/png;base64,{img_base64}")
            plt.close()
        
        # 3. Scatter plot for numeric relationships
        if len(numeric_columns) >= 2:
            plt.figure(figsize=(10, 6))
            
            x_data, y_data = [], []
            for row in data:
                try:
                    x_val = str(row.get(numeric_columns[0], '')).replace(',', '').replace('$', '')
                    y_val = str(row.get(numeric_columns[1], '')).replace(',', '').replace('$', '')
                    if x_val and y_val:
                        x_data.append(float(x_val))
                        y_data.append(float(y_val))
                except:
                    pass
            
            if len(x_data) > 1:
                plt.scatter(x_data, y_data, alpha=0.6, s=50, color=sns.color_palette()[0])
                plt.title(f'{numeric_columns[1]} vs {numeric_columns[0]}', fontsize=14, fontweight='bold')
                plt.xlabel(numeric_columns[0], fontsize=12)
                plt.ylabel(numeric_columns[1], fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # Add trend line if enough points
                if len(x_data) > 3:
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    plt.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
                
                plt.tight_layout()
                
                # Convert to base64
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.read()).decode()
                charts.append(f"data:image/png;base64,{img_base64}")
                plt.close()
        
        # 4. Pie chart for categorical data
        if categorical_cols and len(categorical_cols) > 0:
            plt.figure(figsize=(8, 8))
            col = categorical_cols[0]
            value_counts = df[col].value_counts().head(8)
            
            colors = sns.color_palette("Set3", len(value_counts))
            wedges, texts, autotexts = plt.pie(value_counts.values, 
                                             labels=value_counts.index, 
                                             autopct='%1.1f%%', 
                                             colors=colors,
                                             startangle=90)
            plt.title(f'{col} Distribution', fontsize=14, fontweight='bold')
            
            # Make percentage text more readable
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            charts.append(f"data:image/png;base64,{img_base64}")
            plt.close()
            
    except Exception as e:
        print(f"Error generating charts: {e}")
    
    return charts

# Models
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
app = FastAPI(title="Neo API", version="0.1.0")

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

# Create directories
os.makedirs(".screenshots", exist_ok=True)
app.mount("/screenshots", StaticFiles(directory=".screenshots"), name="screenshots")

# Simple auth
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

# Public endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

@app.get("/screenshot")
async def get_screenshot():
    screenshot_path = ".screenshots/updated_screen.png"
    if os.path.exists(screenshot_path):
        return FileResponse(screenshot_path)
    return JSONResponse(status_code=404, content={"error": "No screenshot available"})

@app.get("/latest_answer")
async def get_latest_answer():
    global last_response
    if last_response:
        return JSONResponse(status_code=200, content=last_response)
    
    return JSONResponse(status_code=200, content={
        "done": "false",
        "answer": "",
        "reasoning": "",
        "agent_name": "Neo",
        "success": "true",
        "blocks": {},
        "status": "Ready",
        "uid": str(uuid.uuid4())
    })

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, authorization: Optional[str] = Header(None)):
    global is_generating, last_response
    
    if is_generating:
        return JSONResponse(status_code=429, content={"error": "Another query is being processed"})
    
    try:
        is_generating = True
        answer = ""
        
        # Handle file uploads
        if request.file_data:
            print(f"Processing file: {request.file_data.name}")
            
            if request.file_data.type.startswith('image/'):
                # Handle image with Gemini Vision
                try:
                    # Extract base64 image data
                    image_data = request.file_data.content
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    
                    # Create prompt with image
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # Decode and prepare image
                    import PIL.Image
                    import io
                    image_bytes = base64.b64decode(image_data)
                    image = PIL.Image.open(io.BytesIO(image_bytes))
                    
                    prompt = f"Please analyze this image and answer the user's question: {request.query}"
                    response = model.generate_content([prompt, image])
                    answer = response.text
                    
                except Exception as e:
                    print(f"Error processing image: {e}")
                    answer = f"I can see you've uploaded an image ({request.file_data.name}). {request.query}"
                    
            elif 'pdf' in request.file_data.type or request.file_data.name.endswith('.pdf'):
                # Handle PDF
                try:
                    import PyPDF2
                    import io
                    
                    # Decode base64 PDF
                    pdf_data = request.file_data.content
                    if ',' in pdf_data:
                        pdf_data = pdf_data.split(',')[1]
                    
                    pdf_bytes = base64.b64decode(pdf_data)
                    pdf_file = io.BytesIO(pdf_bytes)
                    
                    # Extract text from PDF
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    pdf_text = ""
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        pdf_text += page.extract_text() + "\n"
                    
                    # Use Gemini to analyze the PDF content
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    prompt = f"""Analyze this PDF document and answer the user's question.
                    
                    PDF Content (first 5000 characters):
                    {pdf_text[:5000]}
                    
                    User Question: {request.query}
                    
                    Please provide a detailed and helpful response based on the PDF content."""
                    
                    response = model.generate_content(prompt)
                    answer = response.text
                    
                except ImportError:
                    # PyPDF2 not installed, try with basic text extraction
                    answer = "I need PyPDF2 library to read PDF files. Please install it with: pip install PyPDF2"
                except Exception as e:
                    print(f"Error processing PDF: {e}")
                    answer = f"I encountered an error reading the PDF file: {str(e)}. Please try uploading a different format or ensure the PDF is not corrupted."
                    
            elif ('csv' in request.file_data.type or request.file_data.name.endswith('.csv') or 
                  'excel' in request.file_data.type or 'spreadsheet' in request.file_data.type or
                  request.file_data.name.endswith('.xlsx') or request.file_data.name.endswith('.xls')):
                # Handle CSV/Excel with advanced analysis and visualization
                try:
                    import csv
                    import json
                    
                    rows = []
                    file_name = request.file_data.name.split('.')[0]
                    
                    # Check if it's Excel file
                    if request.file_data.name.endswith('.xlsx') or request.file_data.name.endswith('.xls'):
                        try:
                            import pandas as pd
                            # Decode base64 Excel
                            excel_data = request.file_data.content
                            if ',' in excel_data:
                                excel_data = excel_data.split(',')[1]
                            
                            excel_bytes = base64.b64decode(excel_data)
                            excel_file = io.BytesIO(excel_bytes)
                            
                            # Read Excel with pandas
                            df = pd.read_excel(excel_file)
                            rows = df.to_dict('records')
                        except ImportError:
                            answer = "I need pandas library to read Excel files. For now, please use CSV format or install pandas with: pip install pandas openpyxl"
                            rows = []
                    else:
                        # Handle CSV
                        csv_content = request.file_data.content
                        csv_reader = csv.DictReader(io.StringIO(csv_content))
                        rows = list(csv_reader)
                    
                    # Process data if available
                    if rows:
                        columns = list(rows[0].keys())
                        
                        # Detect numeric columns
                        numeric_columns = []
                        for col in columns:
                            try:
                                # Check if first few non-empty values are numeric
                                numeric_count = 0
                                total_count = 0
                                for row in rows[:20]:  # Check first 20 rows
                                    if row[col]:
                                        total_count += 1
                                        try:
                                            float(str(row[col]).replace(',', ''))
                                            numeric_count += 1
                                        except:
                                            pass
                                if total_count > 0 and numeric_count / total_count > 0.8:
                                    numeric_columns.append(col)
                            except:
                                pass
                        
                        # Generate charts as base64 images
                        chart_images = generate_charts_base64(rows, numeric_columns, file_name)
                        
                        # Create enhanced analysis prompt
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        
                        # Limit data for analysis to prevent context overflow
                        sample_data = rows[:100] if len(rows) > 100 else rows
                        
                        # Create table preview
                        table_preview = "| " + " | ".join(columns) + " |\n"
                        table_preview += "| " + " | ".join(["---"] * len(columns)) + " |\n"
                        for row in sample_data[:10]:
                            table_preview += "| " + " | ".join([str(row.get(col, ""))[:20] for col in columns]) + " |\n"
                        
                        # Calculate basic statistics for numeric columns
                        stats_info = ""
                        if numeric_columns:
                            stats_info = "\n**Statistical Summary:**\n"
                            for col in numeric_columns:
                                values = [float(str(row.get(col, 0)).replace(',', '')) for row in sample_data if row.get(col)]
                                if values:
                                    stats_info += f"- {col}: Min={min(values):.2f}, Max={max(values):.2f}, Mean={sum(values)/len(values):.2f}\n"
                        
                        prompt = f"""Analyze this dataset and answer the user's question. Provide comprehensive insights with data tables.

**Dataset Info:**
- File: {request.file_data.name}
- Total rows: {len(rows)}
- Columns: {', '.join(columns)}
- Numeric columns: {', '.join(numeric_columns) if numeric_columns else 'None detected'}
{stats_info}

**Data Preview (first 10 rows):**
{table_preview}

**User Question:** {request.query}

**Response Guidelines:**
1. Start with a clear answer to the user's question
2. Show relevant data in markdown tables
3. Include calculations and statistics where relevant
4. Provide actionable insights
5. Be specific and data-driven

Charts have been generated and will be displayed automatically."""
                        
                        response = model.generate_content(prompt)
                        analysis_text = response.text
                        
                        # Add chart information to response
                        if chart_images:
                            chart_section = f"\n\n## 📊 Visual Analysis\n\nI've generated {len(chart_images)} charts to visualize your data:\n\n"
                            chart_types = ["Distribution", "Category", "Correlation", "Proportion"]
                            
                            for i, img_data in enumerate(chart_images):
                                chart_type = chart_types[i] if i < len(chart_types) else f"Chart {i+1}"
                                chart_section += f"### {i+1}. {chart_type} Analysis\n"
                                chart_section += f"![{chart_type} Chart]({img_data})\n\n"
                            
                            answer = analysis_text + chart_section
                        else:
                            answer = analysis_text
                    else:
                        answer = "The file appears to be empty or improperly formatted."
                        
                except Exception as e:
                    print(f"Error processing data file: {e}")
                    answer = f"I encountered an error processing the file: {str(e)}"
            else:
                answer = f"I've received your file ({request.file_data.name}). {request.query}"
        else:
            # Regular query without file
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(request.query)
            answer = response.text
        
        # Create response
        last_response = {
            "done": "true",
            "answer": answer,
            "reasoning": "",
            "agent_name": "Neo",
            "success": "true",
            "blocks": {},
            "status": "Complete",
            "uid": str(uuid.uuid4())
        }
        
        return QueryResponse(**last_response)
        
    except Exception as e:
        print(f"Error in query processing: {e}")
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
    print("Starting Neo API with fixed image support...")
    uvicorn.run(app, host="0.0.0.0", port=8000)