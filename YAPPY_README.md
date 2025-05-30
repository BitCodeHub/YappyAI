# Yappy - PDF Assistant with AI

**Yappy** is your friendly PDF assistant that helps you analyze documents, extract information, and visualize data - all running locally on your computer for complete privacy.

## Features

### 📄 PDF Analysis
- Upload any PDF file for instant analysis
- Ask questions about the content
- Extract specific information
- Search through documents
- Get summaries and insights

### 📊 Data Visualization
- Upload CSV/Excel files
- Generate beautiful charts automatically
- Create specialized visualizations:
  - Knowledge graphs
  - Flow diagrams
  - Cause & effect diagrams
  - Timelines
  - Heatmaps
  - And more!

### 🔒 Privacy First
- 100% local - your files never leave your computer
- No cloud uploads
- No data tracking
- Complete privacy

## Quick Start

### Easy Launch (No Technical Knowledge Required)

**Mac Users:**
1. Double-click `Yappy.command`

**Windows Users:**
1. Double-click `Yappy.bat`

**All Platforms:**
1. Run `python3 Yappy.py`

The app will:
- ✅ Auto-install any missing components
- ✅ Start the server
- ✅ Open your browser
- ✅ Be ready to use!

## How to Use

1. **Upload a PDF**
   - Click the 📎 button
   - Select your PDF file
   - Yappy will analyze it automatically

2. **Ask Questions**
   - "What's in this document?"
   - "Summarize the main points"
   - "Find information about [topic]"
   - "What does page 5 say?"

3. **Upload Data Files**
   - CSV or Excel files
   - Ask for specific visualizations
   - "Show me a knowledge graph"
   - "Create a timeline"

## Installation for Developers

```bash
# Clone the repository
git clone [repository-url]
cd yappy

# Install dependencies
pip install -r requirements.txt

# Run the API
python api_clean.py
```

## Creating a Deployment Package

To share Yappy with others:

```bash
python create_deployment.py
```

This creates `yappy_pdf_deploy.zip` that includes everything needed.

## System Requirements

- Python 3.8 or higher
- Google Gemini API key (get free at https://makersuite.google.com/app/apikey)
- 4GB RAM recommended
- Any modern web browser

## Troubleshooting

**Port already in use?**
- Run: `python stop_api.py`
- Or use a different port: `python api_clean.py 8001`

**Dependencies missing?**
- Run: `python check_dependencies.py`
- Then: `pip install -r requirements.txt`

**PDF not analyzing?**
- Check the terminal for error messages
- Ensure the PDF isn't corrupted
- Try a different PDF file

## Support

For issues or questions, please open an issue on GitHub or check the documentation files included.

---

Made with ❤️ for easy PDF analysis