#!/bin/bash
# Quick ML Service Deployment Helper
# Guides through deployment to various platforms

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Menu
show_menu() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}ML SERVICE DEPLOYMENT HELPER${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Choose deployment platform:"
    echo ""
    echo "  1) Railway (Recommended - Simple & Fast)"
    echo "  2) Render (Free tier available)"
    echo "  3) Fly.io (Global deployment)"
    echo "  4) Docker (Local/Self-hosted)"
    echo "  5) Local Development"
    echo "  6) Exit"
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
}

# Railway deployment
railway_deploy() {
    echo ""
    echo -e "${YELLOW}RAILWAY DEPLOYMENT${NC}"
    echo ""
    echo "Prerequisites:"
    echo "  ✓ GitHub repository connected"
    echo "  ✓ Railway account created"
    echo ""
    echo "Steps:"
    echo ""
    echo "1. Go to https://railway.app"
    echo "2. Create new project > Deploy from GitHub"
    echo "3. Select your repository"
    echo "4. In Project Settings:"
    echo "   - Root Directory: ml-service"
    echo ""
    echo "5. Add Environment Variables:"
    echo "   - ENVIRONMENT=production"
    echo "   - LOG_LEVEL=INFO"
    echo "   - DATABASE_URL=postgresql://..."
    echo "   - REDIS_URL=redis://..."
    echo "   - PYTHONUNBUFFERED=1"
    echo ""
    echo "6. Deploy by pushing to GitHub:"
    echo "   git push origin main"
    echo ""
    echo -e "${GREEN}Railway will automatically build and deploy!${NC}"
    echo ""
    echo "View your service at: https://your-project-name.railway.app"
    echo "Health check: https://your-project-name.railway.app/health"
    echo ""
}

# Render deployment
render_deploy() {
    echo ""
    echo -e "${YELLOW}RENDER DEPLOYMENT${NC}"
    echo ""
    echo "Prerequisites:"
    echo "  ✓ GitHub repository"
    echo "  ✓ Render account created"
    echo ""
    echo "Steps:"
    echo ""
    echo "1. Go to https://render.com"
    echo "2. Create new Web Service"
    echo "3. Connect GitHub repository"
    echo "4. Configure:"
    echo "   Name: solar-ml-service"
    echo "   Root Directory: ml-service"
    echo "   Runtime: Python 3.10"
    echo "   Build Command: pip install -r requirements.txt"
    echo "   Start Command: python3 run.py"
    echo ""
    echo "5. Environment Variables:"
    echo "   ENVIRONMENT=production"
    echo "   LOG_LEVEL=INFO"
    echo "   PYTHONUNBUFFERED=1"
    echo ""
    echo "6. Create Service"
    echo ""
    echo -e "${GREEN}Render will build and deploy automatically!${NC}"
    echo ""
    echo "Your service URL will be shown on deployment completion"
    echo ""
}

# Fly.io deployment
flyio_deploy() {
    echo ""
    echo -e "${YELLOW}FLY.IO DEPLOYMENT${NC}"
    echo ""
    echo "Prerequisites:"
    echo "  ✓ Fly CLI installed: https://fly.io/docs/hands-on/install/"
    echo "  ✓ Fly account created"
    echo "  ✓ Logged in: fly auth login"
    echo ""
    echo "Steps:"
    echo ""
    echo "1. Install Fly CLI:"
    echo "   curl -L https://fly.io/install.sh | sh"
    echo ""
    echo "2. Login to Fly:"
    echo "   fly auth login"
    echo ""
    echo "3. Initialize from ml-service directory:"
    echo "   cd ml-service"
    echo "   fly launch"
    echo "   (Choose region closest to you)"
    echo ""
    echo "4. Set secrets:"
    echo "   fly secrets set ENVIRONMENT=production"
    echo "   fly secrets set LOG_LEVEL=INFO"
    echo "   fly secrets set DATABASE_URL=\"postgresql://...\""
    echo ""
    echo "5. Deploy:"
    echo "   fly deploy"
    echo ""
    echo -e "${GREEN}Your app is deployed!${NC}"
    echo ""
    echo "View status: fly status"
    echo "View logs: fly logs"
    echo ""
}

# Docker deployment
docker_deploy() {
    echo ""
    echo -e "${YELLOW}DOCKER DEPLOYMENT${NC}"
    echo ""
    echo "For local machine or self-hosted server:"
    echo ""
    echo "1. Build Docker image:"
    echo "   cd ml-service"
    echo "   docker build -t solar-ml-service ."
    echo ""
    echo "2. Run container:"
    echo "   docker run -d \\"
    echo "     --name solar-ml \\"
    echo "     -p 8001:8001 \\"
    echo "     -e ENVIRONMENT=production \\"
    echo "     -e LOG_LEVEL=INFO \\"
    echo "     -v \$(pwd)/logs:/app/logs \\"
    echo "     -v \$(pwd)/models:/app/models \\"
    echo "     solar-ml-service"
    echo ""
    echo "3. Check logs:"
    echo "   docker logs -f solar-ml"
    echo ""
    echo "4. Health check:"
    echo "   curl http://localhost:8001/health"
    echo ""
    echo -e "${GREEN}Container is running!${NC}"
    echo ""
    echo "Access API at: http://localhost:8001/docs"
    echo ""
}

# Local development
local_deploy() {
    echo ""
    echo -e "${YELLOW}LOCAL DEVELOPMENT SETUP${NC}"
    echo ""
    echo "1. Navigate to ml-service:"
    echo "   cd ml-service"
    echo ""
    echo "2. Run startup script:"
    echo "   ./start.sh"
    echo ""
    echo "3. In another terminal, check health:"
    echo "   ./health-check.sh"
    echo ""
    echo "4. Access API:"
    echo "   Browser: http://localhost:8001/docs"
    echo "   Or: curl http://localhost:8001/health"
    echo ""
    echo -e "${GREEN}Service is running locally!${NC}"
    echo ""
    echo "To stop: Press Ctrl+C in service terminal"
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice [1-6]: " choice
    
    case $choice in
        1)
            railway_deploy
            read -p "Press Enter to continue..."
            clear
            ;;
        2)
            render_deploy
            read -p "Press Enter to continue..."
            clear
            ;;
        3)
            flyio_deploy
            read -p "Press Enter to continue..."
            clear
            ;;
        4)
            docker_deploy
            read -p "Press Enter to continue..."
            clear
            ;;
        5)
            local_deploy
            read -p "Press Enter to continue..."
            clear
            ;;
        6)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            sleep 2
            clear
            ;;
    esac
done
