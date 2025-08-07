# Gmail Auto Poller - Cronjob Setup Instructions

## 📋 Overview

This setup allows your system to automatically check Gmail every 5 minutes for new emails and trigger the workflow automatically without manual intervention.

## 🔧 Setup Steps

### 1. Test the Script First

Before setting up the cronjob, test the script manually:

```bash
cd /Users/samuelaudette/Documents/code_projects/agent_inbox_1.12
source venv/bin/activate
python gmail_auto_poller.py
```

### 2. Make Sure LangGraph Server is Running

The auto poller needs the LangGraph server to be running:

```bash
python cli.py langgraph
```

### 3. Setup Cronjob

Open your crontab editor:

```bash
crontab -e
```

Add this line to run the Gmail poller every 5 minutes:

```bash
# Gmail Auto Poller - Check for new emails every 5 minutes
*/5 * * * * /Users/samuelaudette/Documents/code_projects/agent_inbox_1.12/run_gmail_poller.sh >/dev/null 2>&1
```

### 4. Alternative: Longer Intervals

If you prefer different intervals:

```bash
# Every 10 minutes
*/10 * * * * /Users/samuelaudette/Documents/code_projects/agent_inbox_1.12/run_gmail_poller.sh >/dev/null 2>&1

# Every 15 minutes  
*/15 * * * * /Users/samuelaudette/Documents/code_projects/agent_inbox_1.12/run_gmail_poller.sh >/dev/null 2>&1

# Every hour at minute 0
0 * * * * /Users/samuelaudette/Documents/code_projects/agent_inbox_1.12/run_gmail_poller.sh >/dev/null 2>&1
```

### 5. Verify Cronjob is Active

Check your active cronjobs:

```bash
crontab -l
```

## 📊 Monitoring

### Log Files

The auto poller creates several log files for monitoring:

- `gmail_poller.log` - Detailed execution logs
- `gmail_poller_cron.log` - Simple completion timestamps  
- `processed_emails.json` - Track of processed email IDs

### Check Logs

```bash
# View recent activity
tail -f gmail_poller.log

# Check processed emails
cat processed_emails.json
```

## 🔧 How It Works

1. **Gmail Check**: Script authenticates with Gmail API using existing tokens
2. **New Email Detection**: Compares current inbox with `processed_emails.json` 
3. **Avoid Duplicates**: Only processes emails that haven't been seen before
4. **Workflow Trigger**: Sends new emails to LangGraph API for processing
5. **Agent Inbox**: Processed emails appear for human review as usual
6. **State Tracking**: Updates the processed emails list after successful processing

## ⚙️ Configuration

You can modify these settings in `gmail_auto_poller.py`:

```python
MAX_EMAILS_TO_CHECK = 10        # How many recent emails to check
LANGGRAPH_API = "http://127.0.0.1:2024"  # LangGraph server URL
```

## 🛑 Stopping the Cronjob

To stop automatic polling:

```bash
crontab -e
# Comment out or delete the Gmail poller line
```

## 🔍 Troubleshooting

### Cronjob Not Running

1. Check cron service is running: `sudo service cron status` (Linux) or `sudo launchctl list | grep cron` (macOS)
2. Check cron logs: `tail -f /var/log/cron.log` (Linux) or `tail -f /var/log/system.log | grep cron` (macOS)
3. Verify script permissions: `ls -la run_gmail_poller.sh`

### Authentication Issues

1. Make sure `fresh_token.pickle` exists and is valid
2. Re-run OAuth setup: `python simple_oauth_setup.py`
3. Check token file permissions

### LangGraph Connection Issues

1. Ensure LangGraph dev server is running on port 2024
2. Check server logs: `python cli.py langgraph`
3. Verify Agent Inbox is accessible at `http://localhost:3000`

## 🎯 Expected Behavior

With this setup:
- ✅ New emails automatically trigger the workflow
- ✅ Processed emails appear in Agent Inbox for review
- ✅ Human approval still required before sending responses
- ✅ No duplicate processing of the same email
- ✅ Detailed logging for debugging

The system becomes fully automated for **email ingestion** while maintaining **human control** over responses!
