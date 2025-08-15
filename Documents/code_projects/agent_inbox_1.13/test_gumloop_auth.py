#!/usr/bin/env python3
"""
Test Gumloop Google Calendar Authentication
"""
import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

async def test_gumloop_auth():
    """Test and debug Gumloop authentication."""
    load_dotenv()
    
    url = os.environ.get("MCP_SERVER_GOOGLE_AGENDA")
    if not url:
        print("❌ MCP_SERVER_GOOGLE_AGENDA not found in .env")
        return
    
    print(f"✅ Found Gumloop URL: {url[:60]}...")
    
    # Extract user ID from URL for debugging
    if "/gcalendar/" in url:
        user_part = url.split("/gcalendar/")[1].split(":")[0]
        print(f"🔍 Gumloop User ID: {user_part}")
    
    try:
        servers = {
            "google-calendar": {
                "url": url,
                "transport": "sse"
            }
        }
        
        client = MultiServerMCPClient(servers)
        tools = await client.get_tools()
        
        print(f"✅ MCP Connection: SUCCESS - {len(tools)} tools loaded")
        
        # Test actual calendar access
        list_tool = None
        for tool in tools:
            if "list_events" in tool.name:
                list_tool = tool
                break
        
        if list_tool:
            print("🧪 Testing calendar access...")
            try:
                result = await list_tool.ainvoke({
                    "calendar_id": "primary",
                    "max_results": 1
                })
                print("✅ Google Calendar: AUTHENTICATED & WORKING")
                return True
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Google Calendar: AUTHENTICATION FAILED")
                print(f"   Error: {error_msg}")
                
                if "Credentials not found" in error_msg:
                    print("\n🔧 SOLUTION:")
                    print("1. Go to: https://www.gumloop.com/dashboard")
                    print("2. Find your MCP server/workflow")
                    print("3. Connect/Re-authenticate Google Calendar")
                    print("4. Ensure Calendar permissions are granted")
                return False
        else:
            print("❌ No list_events tool found")
            return False
            
    except Exception as e:
        print(f"❌ MCP Connection failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_gumloop_auth())
