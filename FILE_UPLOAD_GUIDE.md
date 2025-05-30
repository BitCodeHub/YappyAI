# File Upload Feature Guide

## Overview
The Neo chatbot now supports file upload functionality, allowing you to upload and analyze various file types including CSV files, images, and more.

## Supported File Types
- **CSV Files** (.csv) - For data analysis and insights
- **Images** (.png, .jpg, .jpeg, .gif) - For visual analysis and chart interpretation
- **Text Files** (.txt) - For text analysis
- **JSON Files** (.json) - For structured data analysis
- **PDF Files** (.pdf) - For document analysis

## How to Use

### 1. Upload a File
- Click the 📁 upload button next to the send button
- Select your file from your computer
- The file will appear above the input field with a preview (for images)

### 2. Ask Questions
Once a file is uploaded, you can ask questions about it:

#### For CSV Files:
- "Analyze this data and show me the main trends"
- "What are the top 5 products by revenue?"
- "Create a visualization showing sales by month"
- "Find any anomalies in the data"

#### For Images:
- "What does this chart show?"
- "Extract the data from this graph"
- "Suggest improvements for this visualization"
- "What insights can you derive from this image?"

#### For Business Data:
- "Summarize the key metrics"
- "What patterns do you see?"
- "Compare this period to the previous one"
- "What actions would you recommend based on this data?"

### 3. Remove Files
- Click the ✕ button next to the file name to remove it
- You can then upload a different file or continue without files

## Example Use Cases

### App Store Reviews Analysis
1. Upload a CSV file containing app reviews
2. Ask: "What are the most common complaints?"
3. Ask: "Show me the sentiment trend over time"
4. Ask: "What features do users love most?"

### Sales Dashboard Analysis
1. Upload an image of your sales dashboard
2. Ask: "What are the key takeaways from this dashboard?"
3. Ask: "Which products are underperforming?"
4. Ask: "Create a better visualization for this data"

### Financial Reports
1. Upload a CSV with financial data
2. Ask: "Calculate the year-over-year growth"
3. Ask: "Identify any unusual transactions"
4. Ask: "Project next quarter's revenue based on this trend"

## Technical Details

### Frontend Changes
- Added file upload state management in React
- Created file preview component for images
- Enhanced UI with file information display
- Integrated file data with API requests

### Backend Changes
- Extended API to accept file uploads
- Added CSV parsing and analysis
- Integrated Google Gemini for image analysis
- Enhanced response generation based on file content

## Installation Requirements
Make sure you have installed the new dependencies:
```bash
pip install pillow google-generativeai
```

## API Changes
The `/query` endpoint now accepts an optional `file_data` object:
```json
{
  "query": "Your question",
  "file_data": {
    "name": "filename.csv",
    "type": "text/csv",
    "content": "base64 encoded content"
  }
}
```

## Security Notes
- Files are processed in memory and not permanently stored
- Base64 encoding is used for secure file transfer
- Temporary files are immediately deleted after processing