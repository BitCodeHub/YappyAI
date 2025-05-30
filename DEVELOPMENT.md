# Development Workflow for Yappy

This guide explains how to continuously develop and deploy features for the Yappy application.

## Quick Start for Development

1. **Clone and Setup**
   ```bash
   git clone <your-repo-url>
   cd agenticSeek-main
   
   # Install backend dependencies
   pip install -r requirements.txt
   
   # Install frontend dependencies
   cd frontend/agentic-seek-front
   npm install
   cd ../..
   ```

2. **Local Development**
   ```bash
   # Terminal 1 - Start backend
   python api.py
   
   # Terminal 2 - Start frontend
   cd frontend/agentic-seek-front
   npm start
   ```

## Continuous Deployment Setup

### 1. Railway Configuration

Your app is configured for automatic deployment to Railway. To set up:

1. **Connect Repository to Railway**
   - Go to [Railway](https://railway.app)
   - Create new project from GitHub repo
   - Railway will automatically detect the Dockerfile

2. **Set Environment Variables in Railway**
   - `NODE_ENV=production`
   - `PORT=8000`
   - `FRONTEND_URL=https://your-app-name.railway.app`

3. **Get Railway Tokens for GitHub Actions**
   - Railway Project Settings → Tokens → Create new token
   - Add to GitHub Secrets:
     - `RAILWAY_TOKEN`: Your Railway token
     - `RAILWAY_SERVICE_ID`: Your Railway service ID

### 2. GitHub Actions

The CI/CD pipeline automatically:
- Builds frontend and backend
- Runs tests (if available)
- Deploys to Railway on push to main branch
- Comments on pull requests

## Development Workflow

### Adding New Features

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/new-feature-name
   ```

2. **Develop Locally**
   - Make changes to frontend (`frontend/agentic-seek-front/src/`)
   - Make changes to backend (`api.py`, `sources/`)
   - Test locally with `python api.py` and `npm start`

3. **Test Changes**
   ```bash
   # Run any available tests
   python -m pytest tests/ -v
   
   # Check frontend build
   cd frontend/agentic-seek-front
   npm run build
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   git push origin feature/new-feature-name
   ```

5. **Create Pull Request**
   - GitHub Actions will automatically run build checks
   - Review changes, merge to main when ready
   - Automatic deployment to Railway on merge

### Hotfixes

For urgent fixes:
```bash
git checkout main
git pull origin main
git checkout -b hotfix/fix-description
# Make changes
git commit -m "fix: urgent fix description"
git push origin hotfix/fix-description
# Create PR and merge immediately
```

## Feature Development Areas

### Frontend Enhancements (`frontend/agentic-seek-front/src/`)
- **App.js**: Main application logic, chat interface
- **App.css**: Styling, animations, responsive design
- **Auth.js**: Login, model selection, API key management

### Backend Enhancements (`api.py`, `sources/`)
- **api.py**: Main FastAPI application, LLM routing
- **sources/agents/**: Agent implementations (browser, code, file, etc.)
- **sources/tools/**: Tool implementations (search, interpreters, etc.)

### Common Enhancement Ideas
- Add new LLM providers
- Improve Yappy character animations
- Add voice input/output
- Implement conversation history
- Add file upload capabilities
- Create mobile-responsive design
- Add dark/light theme toggle
- Implement user preferences storage

## Deployment Monitoring

### Check Deployment Status
1. **Railway Dashboard**: Monitor logs and metrics
2. **GitHub Actions**: View build/deployment status
3. **Application Health**: Visit `https://your-app.railway.app/health`

### Rollback if Needed
```bash
# Revert last commit and redeploy
git revert HEAD
git push origin main
```

## Environment Management

### Local Development
- Uses `http://127.0.0.1:8000` for backend
- Uses `http://localhost:3000` for frontend
- API keys entered by users at runtime

### Production (Railway)
- Single container serving both frontend and backend
- Static files served by FastAPI
- CORS configured for production domain
- API keys managed by users in session storage

## Best Practices

1. **Always test locally** before pushing
2. **Use descriptive commit messages** following conventional commits
3. **Keep API keys secure** - never hardcode in source
4. **Update documentation** when adding new features
5. **Monitor Railway logs** after deployment
6. **Use feature branches** for non-trivial changes

## Troubleshooting

### Common Issues
- **Build fails**: Check dependencies in `requirements.txt` and `package.json`
- **Deployment fails**: Check Railway logs and environment variables
- **API issues**: Verify CORS settings and production URLs
- **Frontend not loading**: Check static file serving in `api.py`

### Getting Help
- Check Railway deployment logs
- Review GitHub Actions build logs
- Test locally to isolate issues
- Verify environment variables match `.env.example`