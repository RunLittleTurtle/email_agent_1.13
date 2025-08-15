#!/usr/bin/env python3
"""
Ambient Email Agent - Main Entry Point
Processes Gmail emails automatically with human-in-the-loop validation
"""

import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime
import uuid

from src.graph.workflow import create_workflow
from src.integrations.gmail import GmailService
from src.models.state import EmailMessage, AgentState
import structlog

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


async def test_workflow():
    """Test the workflow with a sample email"""
    logger.info("Testing ambient email agent workflow")
    
    # Create test email
    test_email = EmailMessage(
        id=str(uuid.uuid4()),
        subject="Meeting Request - Project Discussion",
        body="Hi team,\n\nI'd like to schedule a meeting next Tuesday at 2 PM to discuss the Q1 project roadmap. Please let me know if this works for everyone.\n\nBest regards,\nJohn",
        sender="john.doe@example.com",
        recipients=["team@example.com"],
        timestamp=datetime.now()
    )
    
    # Create initial state
    initial_state = AgentState(
        email=test_email,
        workflow_id=str(uuid.uuid4())
    )
    
    # Create workflow
    workflow = create_workflow()
    
    # Run workflow
    config = {
        "configurable": {
            "thread_id": initial_state.workflow_id
        }
    }
    
    try:
        result = await workflow.ainvoke(initial_state.dict(), config)
        logger.info("Workflow completed successfully")
        logger.info(f"Draft response: {result.get('draft_response', 'No response generated')}")
        return result
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}", exc_info=True)
        raise


async def main():
    """Main entry point for the ambient email agent"""
    logger.info("Starting Ambient Email Agent")
    
    # Verify required environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "LANGSMITH_API_KEY",
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please copy .env.example to .env and fill in your API keys")
        return
    
    # Enable LangSmith tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ambient-email-agent")
    
    try:
        # Run test workflow
        await test_workflow()
        
    except Exception as e:
        logger.error("Failed to run ambient email agent", error=str(e), exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
