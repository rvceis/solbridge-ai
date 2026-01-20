#!/bin/bash
# Deployment script for Hugging Face Spaces

echo "🤗 Deploying SolBridge AI to Hugging Face Spaces"
echo "================================================"

# Check if git is configured
if ! git config user.name > /dev/null; then
    echo "❌ Git user not configured"
    echo "Run: git config --global user.name 'Your Name'"
    echo "     git config --global user.email 'your.email@example.com'"
    exit 1
fi

# Prompt for Hugging Face username
read -p "Enter your Hugging Face username: " HF_USERNAME

if [ -z "$HF_USERNAME" ]; then
    echo "❌ Username cannot be empty"
    exit 1
fi

# Prompt for Space name
read -p "Enter Space name (default: solbridge-ai): " SPACE_NAME
SPACE_NAME=${SPACE_NAME:-solbridge-ai}

echo ""
echo "📋 Deployment Configuration:"
echo "   Username: $HF_USERNAME"
echo "   Space: $SPACE_NAME"
echo "   URL: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
echo ""

read -p "Continue with deployment? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "❌ Deployment cancelled"
    exit 0
fi

# Prepare files for Hugging Face
echo ""
echo "📦 Preparing files..."

# Copy HF README
if [ -f "README_HF.md" ]; then
    cp README_HF.md README.md
    echo "✅ README prepared"
fi

# Use HF Dockerfile
if [ -f "Dockerfile.hf" ]; then
    cp Dockerfile.hf Dockerfile
    echo "✅ Dockerfile prepared"
fi

# Add Hugging Face remote
HF_REPO="https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
echo ""
echo "🔗 Adding Hugging Face remote..."
git remote remove huggingface 2>/dev/null
git remote add huggingface $HF_REPO
echo "✅ Remote added: $HF_REPO"

# Commit changes
echo ""
echo "📝 Committing Hugging Face configuration..."
git add README.md Dockerfile
git commit -m "Configure for Hugging Face Spaces deployment" || echo "No changes to commit"

# Push to Hugging Face
echo ""
echo "🚀 Pushing to Hugging Face Spaces..."
echo "⚠️  You'll need to authenticate with your Hugging Face token"
echo "    Get your token from: https://huggingface.co/settings/tokens"
echo ""

git push huggingface main --force

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully deployed to Hugging Face Spaces!"
    echo ""
    echo "📍 Your Space URL: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
    echo "📖 API Docs: https://$HF_USERNAME-$SPACE_NAME.hf.space/docs"
    echo ""
    echo "⏱️  Building your Space... (this may take 2-5 minutes)"
    echo "    Check progress at: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
else
    echo ""
    echo "❌ Deployment failed"
    echo "    Make sure you have:"
    echo "    1. Created the Space on Hugging Face"
    echo "    2. Authenticated with: git config credential.helper store"
    echo "    3. Used your Hugging Face token as password"
fi
