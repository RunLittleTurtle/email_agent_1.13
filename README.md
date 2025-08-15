# Ambient Email Agent with LangGraph

## Déploiement sur LangGraph Cloud

### 1. Installation de LangGraph CLI (si pas déjà fait)
```bash
pip install -U langgraph-cli
```

### 2. Connexion à LangGraph Cloud
```bash
langgraph cloud login
```

⚠️ **Known Issues**: Human interrupt mechanism needs debugging (workflow completes vs pausing for review)

## 🏗 Architecture

### Implemented Agents
1. **Email Processor**: Parses emails, extracts context (entities, urgency, actions)
2. **Supervisor**: Classifies intent, routes appropriately (simplified for MVP)
3. **Adaptive Writer**: Generates contextual email responses
4. **Human Review**: Interrupt point for Agent Inbox integration

### Placeholder Agents (Future Implementation)
- Calendar Agent (meeting scheduling)
- RAG Agent (document retrieval)
- CRM Agent (contact management)
- Router Agent (approval workflow)

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Update `.env` with your API keys:
```env
OPENAI_API_KEY=your_key_here
LANGSMITH_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ambient-email-agent
```

### 3. Start LangGraph Dev Server
```bash
# Start the development server
langgraph dev

# Server will run on http://127.0.0.1:2024
# Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### 4. Test with Dummy Email
```bash
# Test the workflow with dummy email
python test_dummy_email.py

# Check workflow status
python check_status.py
```

### 5. Configure Agent Inbox
Add inbox with these settings:
- **Assistant/Graph ID**: `email_agent`
- **Deployment URL**: `http://127.0.0.1:2024`
- **Name**: `Ambient Email Agent MVP`

## 📁 Project Structure

```
src/
├── agents/
│   ├── base_agent.py          # Abstract base class
│   ├── email_processor.py     # Email parsing & context extraction
│   ├── supervisor.py          # Intent classification & routing
│   ├── adaptive_writer.py     # Response generation
│   └── router.py              # (placeholder)
├── graph/
│   └── workflow.py            # LangGraph workflow definition
├── models/
│   └── state.py               # Pydantic state models
└── integrations/              # (future Gmail integration)

app.py                         # LangGraph Cloud app definition
main.py                        # Local testing entry point
langgraph.json                 # LangGraph configuration
test_dummy_email.py            # MVP testing script
check_status.py                # Workflow status checker
```

## 🧪 Testing

The MVP includes comprehensive testing:

- **Dummy Email Test**: Sends simple thank you email through workflow
- **Status Checker**: Monitors workflow execution and interrupts
- **LangSmith Traces**: Visual workflow execution tracking
- **Agent Inbox**: Human-in-the-loop testing interface

## 🔧 Dependencies

Core dependencies (latest stable versions):
- `langgraph` (0.6.3) - Workflow orchestration
- `langchain` - LLM framework
- `langchain-openai` - OpenAI integration
- `langsmith` - Tracing and monitoring
- `pydantic` - Data validation
- `structlog` - Structured logging

## 📊 Current Capabilities

✅ **Email Processing**: Parses subject, body, sender, recipients  
✅ **Context Extraction**: Key entities, urgency, requested actions  
✅ **Intent Classification**: Routes based on email content  
✅ **Response Generation**: Creates appropriate, contextual replies  
✅ **LangSmith Integration**: Full tracing and visualization  
✅ **API Integration**: Works with LangGraph dev server  

## 🎯 Next Steps

1. **Fix Human Interrupt**: Debug `interrupt_after` configuration
2. **Agent Inbox Integration**: Ensure human-in-the-loop appears in inbox
3. **Implement Missing Agents**: Calendar, RAG, CRM agents
4. **Gmail Integration**: Connect to actual Gmail API
5. **Production Deployment**: Move from dev server to production

## 🤝 Contributing

This is an MVP focused on core functionality. Future enhancements welcome!
