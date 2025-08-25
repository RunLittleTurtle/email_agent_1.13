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
# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
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

### 4. Google Authentication
```bash
# Run OAuth setup (will open browser)
python cli.py setup-oauth
```

### 5. Start Services
```bash
# Start everything
python cli.py start

# Or individually:
python cli.py langgraph  # Backend API
python cli.py inbox      # Frontend UI
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

**Python Import Errors:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**Node.js Issues:**
```bash
cd agent-inbox
rm -rf node_modules yarn.lock
yarn install
```

**Gmail Authentication:**
```bash
python cli.py setup-oauth
```

**Port Conflicts:**
```bash
# Kill existing processes
python cli.py stop
python cli.py start
```
