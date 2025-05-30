# PDF Upload Feature Guide

## Overview
The Neo chatbot now supports comprehensive PDF file analysis. When you upload a PDF file, it will automatically extract and analyze the content, allowing you to ask questions about the document.

## Features

### 📄 PDF Analysis
- **Automatic Text Extraction**: Extracts all text content from PDF files
- **Metadata Extraction**: Retrieves document properties (title, author, creation date, etc.)
- **Page-by-Page Analysis**: Access content from specific pages
- **Document Statistics**: Word count, character count, total pages
- **Persistent Context**: The PDF remains in memory for follow-up questions

### 📊 Data File Visualization
- **CSV/Excel Support**: Upload data files for automatic analysis
- **Smart Visualizations**: Ask for specific chart types:
  - Knowledge graphs
  - Flow diagrams
  - Cause & effect diagrams (Fishbone)
  - Hierarchy diagrams
  - Timelines
  - Correlation heatmaps
  - Sankey flow diagrams

## How to Use

### Starting the API
```bash
# From the project directory
python start_neo_api.py
# OR
python api_clean.py
```

### Uploading PDFs
1. Click the 📎 button in the chat interface
2. Select your PDF file
3. The file will be automatically analyzed
4. Ask any questions about the content

### Example Questions for PDFs
- "Summarize this PDF"
- "What are the main points?"
- "Search for [keyword] in the document"
- "What does page 3 say about [topic]?"
- "Extract all statistics mentioned"
- "What are the key findings?"

### Example Requests for Data Visualization
- "Show me a knowledge graph of the relationships"
- "Create a flow diagram of the process"
- "Generate a cause and effect diagram"
- "Display this as a timeline"
- "Show correlation heatmap"

## Troubleshooting

### PDF Not Being Analyzed
If you see "I've received your file" instead of analysis:
1. Check the console/terminal for error messages
2. Ensure pypdf is installed: `pip install pypdf`
3. Try re-uploading the file

### Common Issues
- **Large PDFs**: Files over 10MB may take longer to process
- **Scanned PDFs**: Image-based PDFs without text layers won't extract text
- **Encrypted PDFs**: Password-protected files cannot be read

## Technical Details

### File Processing Flow
1. File upload triggers base64 encoding in frontend
2. API receives file data with name, type, and content
3. PDF detection based on file extension and MIME type
4. Text extraction using pypdf library
5. Content stored in memory for session
6. AI analysis using Google Gemini API

### API Endpoints
- `POST /query` - Main endpoint accepting file_data parameter
- `GET /pdf_info` - Check if PDF is loaded
- `GET /clear_pdf` - Clear PDF from memory

## Requirements
- Python 3.8+
- pypdf library
- google-generativeai
- matplotlib, seaborn (for visualizations)
- pandas (for data analysis)