# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgenticSeek is a 100% local alternative to Manus AI - a voice-enabled AI assistant that autonomously browses the web, writes code, and plans tasks while keeping all data on your device. It uses a multi-agent architecture with specialized agents (Casual, Coder, File, Browser, Planner) and supports both local LLM providers (Ollama, LM Studio) and remote APIs.

## Common Development Commands

### Installation and Setup
```bash
# Create virtual environment (Python 3.10 required)
python3 -m venv agentic_seek_env
source agentic_seek_env/bin/activate  # Linux/macOS
# or: agentic_seek_env\Scripts\activate  # Windows

# Install dependencies
./install.sh  # Linux/macOS
# or: ./install.bat  # Windows

# Start services (Docker, SearxNG, Redis, Frontend)
sudo ./start_services.sh  # macOS
# or: start ./start_services.cmd  # Windows
```

### Running the Application
```bash
# CLI mode
python3 cli.py

# Web interface mode
python3 api.py  # Then visit http://localhost:3000
```

### Running Tests
```bash
# Run all Python tests
python -m unittest discover tests/

# Run specific test files
python -m unittest tests.test_browser_agent_parsing
python -m unittest tests.test_memory
python -m unittest tests.test_provider
python -m unittest tests.test_searx_search

# Frontend tests
cd frontend/agentic-seek-front
npm test
```

### Frontend Development
```bash
cd frontend/agentic-seek-front
npm start  # Development server
npm build  # Production build
```

## Architecture Overview

### Multi-Agent System
The project uses specialized agents orchestrated by a router system:
- **Planner Agent** (`sources/agents/planner_agent.py`): Decomposes complex tasks into steps
- **Browser Agent** (`sources/agents/browser_agent.py`): Web browsing and form filling using Selenium
- **Code Agent** (`sources/agents/code_agent.py`): Code generation and execution
- **File Agent** (`sources/agents/file_agent.py`): File system operations
- **Casual Agent** (`sources/agents/casual_agent.py`): General conversation
- **MCP Agent** (`sources/agents/mcp_agent.py`): Model Context Protocol support

### Core Components
- **Router** (`sources/router.py`): Intelligent agent selection using local LLM classification
- **LLM Provider** (`sources/llm_provider.py`): Unified interface for multiple LLM providers
- **Memory System** (`sources/memory.py`): Session persistence and context management
- **Browser Automation** (`sources/browser.py`): Selenium-based web interaction with stealth mode
- **Tool System** (`sources/tools/`): Interpreters for Python, C, Go, Java, Bash

### Configuration
Main configuration is in `config.ini`:
- `is_local`: Toggle between local and API providers
- `provider_name`: LLM provider (ollama, lm-studio, openai, etc.)
- `provider_model`: Specific model to use
- `work_dir`: Workspace directory for file operations
- `headless_browser`: Browser visibility toggle
- `languages`: Supported languages for routing

### LLM Server
For remote model hosting, use the dedicated server in `llm_server/`:
```bash
cd llm_server/
python3 app.py --provider ollama --port 3333
```

## Key Implementation Details

### Agent Communication
- Agents use structured prompts in `prompts/base/` and `prompts/jarvis/`
- Each agent has specific output format requirements enforced via prompts
- Router uses local classification models for agent selection

### Browser Automation
- Uses undetected-chromedriver for stealth mode
- JavaScript injection for form detection and interaction
- Safety mechanisms to prevent malicious sites

### Security Considerations
- All code execution happens in controlled interpreters
- File operations restricted to configured work directory
- Browser includes safety script injection
- JWT-based authentication protects API endpoints
- User passwords stored with bcrypt hashing

### Testing
- Tests use Python's unittest framework
- Mock providers for testing LLM interactions
- Browser agent parsing tests for command extraction

## Current Development Notes
- 2 is running. how do i do #3
- ALWAYS provide git commands after completing any file changes or tasks