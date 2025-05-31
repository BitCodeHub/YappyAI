# 🐕 Yappy AI Complete - All Features Guide

## Overview

Yappy AI Complete includes ALL features from the local AgenticSeek app:

### ✅ All Agent Types
- **Casual Agent** - General conversation with web search
- **Coder Agent** - Code generation and execution (Python, Bash, Java, Go, C)
- **File Agent** - File operations and document analysis
- **Planner Agent** - Task breakdown and multi-agent coordination
- **Browser Agent** - Web automation (coming soon)
- **MCP Agent** - MCP protocol tools (coming soon)

### ✅ All Tools
- **Code Interpreters** - Execute Python, Bash, Java, Go, and C code
- **File Operations** - Search, read, and analyze files
- **Web Search** - Real-time web information retrieval
- **PDF Analysis** - Extract and analyze PDF content
- **Task Planning** - Break down complex tasks
- **Memory Compression** - Manage long conversations

### ✅ Advanced Features
- **Multi-LLM Support** - OpenAI, Anthropic, Google, Groq
- **Agent Routing** - Intelligent agent selection
- **Conversation History** - Persistent chat storage
- **File Uploads** - Process documents and images
- **User Statistics** - Track usage and performance
- **Feature Flags** - Enable/disable features per user

## Quick Start

### 1. Local Development

```bash
# Install dependencies
pip install -r requirements_v2.txt

# Set environment variable for database (optional)
export DATABASE_URL="postgresql://user:pass@localhost/yappy"

# Run the application
python app_complete_v2.py
```

Visit http://localhost:8000

### 2. Deploy to Render

1. Fork this repository
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Use these settings:
   - **Build Command**: `pip install -r requirements_v2.txt`
   - **Start Command**: `python app_complete_v2.py`
   - **Add PostgreSQL database** (Render will set DATABASE_URL automatically)

### 3. Deploy with Docker

```bash
# Build the image
docker build -f Dockerfile.complete -t yappy-complete .

# Run with PostgreSQL
docker run -p 8000:8000 \
  -e DATABASE_URL="postgresql://user:pass@db/yappy" \
  yappy-complete
```

## Usage Examples

### 1. Code Generation and Execution

Select the **Coder Agent** and ask:
```
Write a Python function to calculate fibonacci numbers and test it
```

Yappy will:
- Generate the code
- Execute it automatically
- Show the results

### 2. File Operations

Select the **File Agent** and ask:
```
Find all Python files in the current directory
```

Or upload a file and ask:
```
Analyze this PDF and summarize the key points
```

### 3. Task Planning

Select the **Planner Agent** for complex tasks:
```
Help me build a web scraper that extracts product prices, saves them to a CSV, and creates a price trend chart
```

Yappy will:
- Break down the task into steps
- Identify which agents to use
- Guide you through implementation

### 4. Web Search

Any agent can search the web when needed:
```
What's the current weather in San Francisco?
What are the latest AI news?
```

## API Endpoints

### Authentication
- `POST /auth/register` - Create account
- `POST /auth/login` - Login

### Chat
- `POST /api/chat` - Send message with agent selection
- `GET /api/conversations` - List conversations
- `GET /api/agents/available` - Get available agents

### User Management
- `GET /api/user/profile` - Get profile
- `POST /api/user/api-key` - Update API keys
- `GET /api/stats` - Get usage statistics

### Files
- `POST /api/upload` - Upload files

### Task Planning
- `POST /api/task/plan` - Create task plan

## Configuration

### Environment Variables
- `DATABASE_URL` - PostgreSQL connection string
- `PORT` - Server port (default: 8000)

### Adding API Keys

After login, add your API keys in the profile section:
- OpenAI API Key
- Anthropic API Key
- Google Gemini API Key
- Groq API Key

## Security Notes

- All API keys are stored encrypted per user
- Code execution is sandboxed
- File operations are restricted to safe paths
- No dangerous shell commands allowed

## Roadmap

### Coming Soon
- [ ] Browser automation with Selenium
- [ ] Voice input/output (TTS/STT)
- [ ] MCP protocol integration
- [ ] Real-time collaboration
- [ ] Plugin system

## Troubleshooting

### Database Connection Issues
If you see database errors:
1. Ensure PostgreSQL is running
2. Check DATABASE_URL format
3. For SQLite fallback, leave DATABASE_URL unset

### Code Execution Not Working
1. Ensure Python/Bash are installed
2. Check file permissions
3. Review security settings

### File Upload Issues
1. Check file size limits
2. Ensure upload directory exists
3. Verify file type support

## Support

For issues or questions:
- GitHub Issues: [Create an issue](https://github.com/yourusername/yappy-ai/issues)
- Documentation: This guide
- Community: Coming soon!

---

🐕 **Woof!** Yappy AI is tail-waggingly excited to help you with ALL the features!