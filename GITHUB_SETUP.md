# 🚀 GitHub Setup Guide for Yappy

Follow these steps to push your Yappy app to GitHub:

## Step 1: Initialize Git (in Terminal)

```bash
cd /Users/jimmylam/Downloads/agenticSeek-main
git init
```

## Step 2: Add all files to Git

```bash
git add .
git commit -m "feat: initial commit - Yappy AI assistant with cloud deployment"
```

## Step 3: Create GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Create a new repository with these settings:
   - **Repository name**: `yappy-ai` (or your preferred name)
   - **Description**: "Yappy - A friendly AI assistant with multi-LLM support"
   - **Public** or **Private**: Your choice
   - **DO NOT** initialize with README, .gitignore, or license (we already have them)

## Step 4: Connect and Push to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
# Add your GitHub repository as remote origin
git remote add origin https://github.com/YOUR_USERNAME/yappy-ai.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 5: Verify Upload

- Go to your GitHub repository page
- You should see all your files uploaded
- Check that `.gitignore` is working (no .env, logs, or node_modules)

## Step 6: Enable GitHub Actions (Optional)

If you want automatic deployments:
1. Go to Settings → Actions → General
2. Under "Workflow permissions", select "Read and write permissions"
3. Save

## Troubleshooting

### If you get authentication errors:
- You may need to use a Personal Access Token instead of password
- Go to GitHub Settings → Developer settings → Personal access tokens
- Generate a new token with "repo" permissions
- Use the token as your password when prompted

### If you see large file warnings:
- Make sure node_modules and build folders are not being tracked
- Run: `git rm -r --cached frontend/agentic-seek-front/node_modules` if needed
- Then commit again

## Next Steps

Once pushed to GitHub:
1. Go to [Railway](https://railway.app)
2. Create new project → Deploy from GitHub repo
3. Select your repository
4. Railway will automatically deploy!

---

**Need help?** The most common issues are:
- Forgetting to NOT initialize the GitHub repo with README
- Authentication issues (use Personal Access Token)
- Large files (check .gitignore is working)