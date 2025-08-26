#!/bin/bash

# =============================================================================
# Agent Inbox - Automated Setup Script
# =============================================================================
# This script sets up the complete Agent Inbox environment in one command

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }

# Banner
echo -e "${BLUE}"
echo "🤖 Agent Inbox - Automated Setup"
echo "=================================="
echo -e "${NC}"

# Check if running on supported OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    info "Detected macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    info "Detected Linux"
else
    error "Unsupported OS: $OSTYPE"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system dependencies
info "Checking system dependencies..."

# Check Python 3.13+
if ! command_exists python3.13 && ! command_exists python3; then
    error "Python 3.13+ not found. Please install:"
    if [[ $OS == "macos" ]]; then
        echo "  brew install python@3.13"
    else
        echo "  sudo apt install python3.13 python3.13-venv"
    fi
    exit 1
fi

# Check Node.js
if ! command_exists node; then
    error "Node.js not found. Please install:"
    if [[ $OS == "macos" ]]; then
        echo "  brew install node@20"
    else
        echo "  sudo apt install nodejs npm"
    fi
    exit 1
fi

# Check Yarn
if ! command_exists yarn; then
    warning "Yarn not found. Installing..."
    if [[ $OS == "macos" ]]; then
        brew install yarn
    else
        npm install -g yarn
    fi
fi

success "System dependencies verified"

# Get Python executable
if command_exists python3.13; then
    PYTHON="python3.13"
elif command_exists python3; then
    PYTHON="python3"
else
    error "No suitable Python found"
    exit 1
fi

info "Using Python: $($PYTHON --version)"

# Create virtual environment
info "Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    $PYTHON -m venv .venv
    success "Virtual environment created"
else
    warning "Virtual environment already exists"
fi

# Activate virtual environment
info "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
info "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
info "Installing Python dependencies..."
pip install -r requirements.txt

# Ensure langgraph-api is installed (critical for server)
info "Installing langgraph-api with inmem support..."
pip install -U "langgraph-cli[inmem]"

# Verify core packages work
info "Verifying core package imports..."
python -c "import langgraph; import langchain; import psutil; print('✅ Core packages working')" || {
    error "Core package import failed. Reinstalling requirements..."
    pip install --force-reinstall -r requirements.txt
}

success "Python dependencies installed and verified"

# Setup environment file
info "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    success "Created .env from template"
    warning "Please edit .env with your API keys before proceeding"
else
    info ".env already exists"
fi

# Install Node.js dependencies
info "Installing Node.js dependencies..."
cd agent-inbox
yarn install

# Create required Next.js utility files if missing
info "Setting up Next.js utility files..."
mkdir -p src/lib

# Create utils.ts if missing
if [ ! -f "src/lib/utils.ts" ]; then
    info "Creating src/lib/utils.ts..."
    cat > src/lib/utils.ts << 'EOF'
import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
EOF
    success "Created src/lib/utils.ts"
fi

# Check if client.ts exists, if not warn user
if [ ! -f "src/lib/client.ts" ]; then
    warning "src/lib/client.ts missing - will be created during first run"
fi

success "Node.js dependencies installed and configured"
cd ..

# Check for required API keys
info "Checking environment configuration..."
if [ -f ".env" ]; then
    if grep -q "your_openai_api_key_here" .env; then
        warning "Please add your OPENAI_API_KEY to .env"
    fi
    if grep -q "your_langsmith_api_key_here" .env; then
        warning "Please add your LANGSMITH_API_KEY to .env"
    fi
fi

# Make CLI executable
chmod +x cli.py

echo ""
success "🎉 Setup complete!"
echo ""
info "Next steps:"
echo "1. Edit .env with your API keys (OPENAI_API_KEY, LANGSMITH_API_KEY)"
echo "2. Run OAuth setup: python cli.py setup-oauth"
echo "3. Start services: python cli.py start"
echo "4. Access UI: http://localhost:3000"
echo ""
info "Quick start commands:"
echo "  source .venv/bin/activate  # Activate environment"
echo "  python cli.py --help       # See available commands"
echo "  python cli.py status       # Check service status"
echo "  python cli.py gmail        # Test Gmail integration"
echo ""
success "Agent Inbox is ready to use!"
