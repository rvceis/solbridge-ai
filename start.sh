#!/bin/bash
# ML Service Startup Script
# Handles virtual environment, dependencies, and service startup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}SOLAR SHARING ML SERVICE - STARTUP SCRIPT${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}1️⃣  Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $PYTHON_VERSION"

# Check if Python 3.10+
REQUIRED_VERSION="3.10"
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo -e "${RED}✗ Python 3.10+ required (found: $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Check virtual environment
echo -e "${YELLOW}2️⃣  Checking virtual environment...${NC}"
VENV_DIR="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}   Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${YELLOW}3️⃣  Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Install/upgrade dependencies
echo -e "${YELLOW}4️⃣  Installing dependencies...${NC}"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -q -r "$PROJECT_ROOT/requirements.txt" 2>&1 | grep -E "Successfully installed|already satisfied|ERROR" || true
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Check environment variables
echo -e "${YELLOW}5️⃣  Checking environment configuration...${NC}"
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${GREEN}✓ .env file found${NC}"
    source "$PROJECT_ROOT/.env"
else
    echo -e "${YELLOW}⚠  No .env file found, using defaults${NC}"
fi
echo ""

# Create logs directory
echo -e "${YELLOW}6️⃣  Setting up logging...${NC}"
mkdir -p "$PROJECT_ROOT/logs"
echo -e "${GREEN}✓ Log directory ready: $PROJECT_ROOT/logs${NC}"
echo ""

# Display startup info
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}SERVICE CONFIGURATION${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "Environment:      ${ENVIRONMENT:-development}"
echo -e "Host:             ${ML_HOST:-0.0.0.0}"
echo -e "Port:             ${ML_PORT:-8001}"
echo -e "Log Level:        ${LOG_LEVEL:-INFO}"
echo -e "Log Directory:    $PROJECT_ROOT/logs"
echo -e "Python:           $PYTHON_VERSION"
echo -e "Virtual Env:      $VENV_DIR"
echo ""

# Start service
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 STARTING ML SERVICE${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

cd "$PROJECT_ROOT"
exec python3 run.py
