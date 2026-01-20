# PowerShell deployment script for Hugging Face Spaces
# Run this on Windows

Write-Host "🤗 Deploying SolBridge AI to Hugging Face Spaces" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check git configuration
try {
    $gitUser = git config user.name
    if (-not $gitUser) {
        throw "Git user not configured"
    }
} catch {
    Write-Host "❌ Git user not configured" -ForegroundColor Red
    Write-Host "Run: git config --global user.name 'Your Name'" -ForegroundColor Yellow
    Write-Host "     git config --global user.email 'your.email@example.com'" -ForegroundColor Yellow
    exit 1
}

# Get Hugging Face username
$HF_USERNAME = Read-Host "Enter your Hugging Face username"
if (-not $HF_USERNAME) {
    Write-Host "❌ Username cannot be empty" -ForegroundColor Red
    exit 1
}

# Get Space name
$SPACE_NAME = Read-Host "Enter Space name (press Enter for default: solbridge-ai)"
if (-not $SPACE_NAME) {
    $SPACE_NAME = "solbridge-ai"
}

Write-Host ""
Write-Host "📋 Deployment Configuration:" -ForegroundColor Green
Write-Host "   Username: $HF_USERNAME" -ForegroundColor White
Write-Host "   Space: $SPACE_NAME" -ForegroundColor White
Write-Host "   URL: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" -ForegroundColor White
Write-Host ""

$CONFIRM = Read-Host "Continue with deployment? (y/n)"
if ($CONFIRM -ne "y") {
    Write-Host "❌ Deployment cancelled" -ForegroundColor Red
    exit 0
}

# Prepare files
Write-Host ""
Write-Host "📦 Preparing files..." -ForegroundColor Cyan

# Copy HF README
if (Test-Path "README_HF.md") {
    Copy-Item "README_HF.md" "README.md" -Force
    Write-Host "✅ README prepared" -ForegroundColor Green
}

# Use HF Dockerfile
if (Test-Path "Dockerfile.hf") {
    Copy-Item "Dockerfile.hf" "Dockerfile" -Force
    Write-Host "✅ Dockerfile prepared" -ForegroundColor Green
}

# Add Hugging Face remote
$HF_REPO = "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
Write-Host ""
Write-Host "🔗 Adding Hugging Face remote..." -ForegroundColor Cyan

git remote remove huggingface 2>$null
git remote add huggingface $HF_REPO
Write-Host "✅ Remote added: $HF_REPO" -ForegroundColor Green

# Commit changes
Write-Host ""
Write-Host "📝 Committing Hugging Face configuration..." -ForegroundColor Cyan
git add README.md Dockerfile
git commit -m "Configure for Hugging Face Spaces deployment" 2>$null

# Push to Hugging Face
Write-Host ""
Write-Host "🚀 Pushing to Hugging Face Spaces..." -ForegroundColor Cyan
Write-Host "⚠️  You'll need to authenticate with your Hugging Face token" -ForegroundColor Yellow
Write-Host "    Get your token from: https://huggingface.co/settings/tokens" -ForegroundColor Yellow
Write-Host ""

git push huggingface main --force

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Successfully deployed to Hugging Face Spaces!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📍 Your Space URL: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" -ForegroundColor Cyan
    Write-Host "📖 API Docs: https://$HF_USERNAME-$SPACE_NAME.hf.space/docs" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "⏱️  Building your Space... (this may take 2-5 minutes)" -ForegroundColor Yellow
    Write-Host "    Check progress at: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "❌ Deployment failed" -ForegroundColor Red
    Write-Host "    Make sure you have:" -ForegroundColor Yellow
    Write-Host "    1. Created the Space on Hugging Face" -ForegroundColor White
    Write-Host "    2. Used your Hugging Face token as password when prompted" -ForegroundColor White
    Write-Host "    3. Selected 'Docker' as SDK when creating the Space" -ForegroundColor White
}
