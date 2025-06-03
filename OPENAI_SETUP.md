# OpenAI API Setup Instructions

## Important: API Key Configuration

All agents (except Code Agent) now use OpenAI API by default. The Code Agent continues to use Claude API for enhanced coding capabilities.

**Note**: You provided an OpenAI API key. Please set it as an environment variable on your deployment server for security reasons. Do not commit it to the repository.

## Setup Instructions

1. **Environment Variable Method (Recommended)**
   
   Set the following environment variable on your server:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

2. **Docker Method**
   
   Add to your docker-compose.yml:
   ```yaml
   environment:
     - OPENAI_API_KEY=your-openai-api-key-here
   ```

3. **.env File Method**
   
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your-openai-api-key-here
   ```

## Agent Configuration

- **All Agents** (Browser, Planning, Research, File, Casual, Travel): Use OpenAI API
- **Code Agent**: Uses Claude API (requires CLAUDE_API_KEY environment variable)

## Fallback Behavior

If a user hasn't configured their OpenAI API key in the settings:
1. The system will first check for the OPENAI_API_KEY environment variable
2. If not found, the user will be prompted to add their API key in settings

## Security Note

Never commit API keys directly to the repository. Always use environment variables or secure secret management systems.