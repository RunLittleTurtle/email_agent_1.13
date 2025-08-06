"""
LangGraph Cloud App Definition
This file is used by LangGraph Cloud to load the graph
"""

import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Now we can import and create the graph
from src.graph.workflow import create_workflow

# Create the graph instance for LangGraph Cloud
graph = create_workflow()
