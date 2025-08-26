# 🚀 Agent Inbox - One-Command Setup

Complete email agent workflow with Gmail integration and calendar management.

## Prerequisites

### System Dependencies (Required)

```bash
# macOS (using Homebrew)
brew install python@3.13 node@20 yarn git

# Ubuntu/Debian
sudo apt update
sudo apt install python3.13 python3.13-venv nodejs npm yarn git

# Windows (using Chocolatey)
choco install python nodejs yarn git
```

**Required Versions:**
- **Python**: 3.13+ (for type hints and latest features)
- **Node.js**: 18+ (for Next.js 15.5.0)
- **Yarn**: 1.22+ (package manager)
- **Git**: Latest

## 🎯 One-Command Setup

```bash
# Clone and setup everything
git clone <your-repo-url> agent_inbox
cd agent_inbox
./setup.sh
```

## Manual Setup (If Automated Fails)

### 1. Python Environment
```bash
# Create virtual environment (IMPORTANT: Use .venv not venv)
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies (includes langgraph-api for server)
pip install --upgrade pip
pip install -r requirements.txt

# Verify core packages are installed
python -c "import langgraph; import langchain; import psutil; print('✅ Core packages imported successfully')"
```

### 2. Node.js Dependencies
```bash
# Navigate to frontend
cd agent-inbox

# Install dependencies
yarn install
```

### 3. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys:
# - OPENAI_API_KEY (required)
# - LANGSMITH_API_KEY (required)
# - Other keys as needed
```

### 4. Agent Inbox UI Setup (Next.js)
```bash
# Navigate to frontend and install dependencies
cd agent-inbox
yarn install

# Create required utility files (if missing)
mkdir -p src/lib
# utils.ts and client.ts should be created automatically during setup
```

### 5. Google Authentication Setup
```bash
# IMPORTANT: OAuth requires credentials.json file
# If missing, run the credential setup first:
python simple_oauth_setup.py  # This will open browser for authentication

# Alternative: Use CLI setup (may require credentials.json)
python cli.py setup-oauth
```

### 6. Start Services
```bash
# Start everything (will auto-detect and use available ports)
python cli.py start

# Or individually:
python cli.py langgraph  # Backend API
python cli.py inbox      # Frontend UI

# Check status
python cli.py status
```

## 🔧 API Keys Required

Get these API keys before running setup:

1. **OpenAI API Key**
   - Visit: https://platform.openai.com/api-keys
   - Required for AI email processing

2. **LangSmith API Key**
   - Visit: https://smith.langchain.com/
   - Required for monitoring and debugging

3. **Google OAuth** (Auto-configured)
   - Handled by `python cli.py setup-oauth`

## 🚦 Verification

After setup, verify everything works:

```bash
# Check service status
python cli.py status

# Test Gmail integration
python cli.py gmail

# Access UI
open http://localhost:3000
```

## 📁 Project Structure

```
agent_inbox/
├── .venv/                  # Python virtual environment
├── agent-inbox/           # Next.js frontend
├── src/                   # Python backend
├── CLI/                   # Command reference
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── .env                  # Your configuration (git-ignored)
└── setup.sh              # Automated setup script
```

## 🔍 Troubleshooting

### Common Setup Issues

**❌ "Module not found: psutil"**
```bash
# Make sure you're in the correct virtual environment
source .venv/bin/activate  # NOT venv/bin/activate
pip install -r requirements.txt
```

**❌ "Required package 'langgraph-api' is not installed"**
```bash
# Install with inmem flag
source .venv/bin/activate
pip install -U "langgraph-cli[inmem]"
```

**❌ Next.js "Module not found: '@/lib/utils'" or "@/lib/client"**
```bash
cd agent-inbox
mkdir -p src/lib

# Create utils.ts
cat > src/lib/utils.ts << 'EOF'
import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
EOF

# Install dependencies if missing
yarn install
```

**❌ OAuth "credentials.json not found"**
```bash
# Create credentials.json from your Google OAuth app
# Or use the simple OAuth setup that creates it automatically
python simple_oauth_setup.py
```

**❌ Virtual Environment Issues**
```bash
# Remove old environment and recreate
rm -rf venv .venv
python3.13 -m venv .venv  # Use .venv NOT venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**❌ Port Conflicts**
```bash
# Kill existing processes
python cli.py stop
pkill -f "next dev"
pkill -f "langgraph"
python cli.py start
```

**❌ Node.js/Yarn Issues**
```bash
cd agent-inbox
rm -rf node_modules yarn.lock .next
yarn install
```
