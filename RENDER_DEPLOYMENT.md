# Render Deployment Guide

## Environment Variables Setup

The application requires API keys to be configured as environment variables in Render.

### Required Environment Variables:

1. **OPENAI_API_KEY** (Required)
   - Used by all agents except Code Agent
   - Without this, you'll get: "Server configuration error: No API key configured for openai"
   - Get your key from: https://platform.openai.com/api-keys

2. **CLAUDE_API_KEY** (Required for Code Agent)
   - Used by the Code Agent for enhanced code generation
   - Get your key from: https://console.anthropic.com/

### How to Add Environment Variables in Render:

1. Log in to your [Render Dashboard](https://dashboard.render.com/)
2. Select your web service
3. Click on "Environment" in the left sidebar
4. Click "Add Environment Variable"
5. Add each key:
   - Key: `OPENAI_API_KEY`
   - Value: `your-openai-api-key-here`
6. Click "Save Changes"
7. Your service will automatically redeploy with the new environment variables

### Optional Environment Variables:

- `GEMINI_API_KEY` - For Google Gemini support
- `GROQ_API_KEY` - For Groq support
- `SEARXNG_BASE_URL` - Custom search instance (defaults to public instance)

### Troubleshooting:

If you see "Server configuration error: No API key configured":
1. Check that the environment variable name is exactly `OPENAI_API_KEY`
2. Make sure there are no extra spaces in the key
3. Wait for the service to fully redeploy after adding variables
4. Check the Render logs for confirmation that keys are loaded

### Security Note:

Never commit API keys to your repository. Always use environment variables for sensitive data.