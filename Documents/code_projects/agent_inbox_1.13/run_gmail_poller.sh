#!/bin/bash
# Gmail Auto Poller - Cronjob Wrapper Script
# This script activates the virtual environment and runs the Gmail poller

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to project directory
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Run the Gmail poller
python gmail_auto_poller.py

# Log completion
echo "$(date): Gmail poller completed" >> gmail_poller_cron.log
