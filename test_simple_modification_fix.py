#!/usr/bin/env python3
"""
Simple test to verify calendar modification detection logic without API calls.
"""

def test_modification_keywords():
    """Test that our modification detection keywords cover the key use cases"""
    
    # Test cases that should be detected as modifications
    modification_phrases = [
        "change the event",
        "change the meeting", 
        "modify the booking",
        "update the booking",
        "reschedule",
        "move the meeting",
        "new date",
        "different time",
        "cancel and reschedule"
    ]
    
    # Sample email bodies that should trigger modification detection
    test_emails = [
        "ok change the event in the calendar please Le ven. 22 août 2025 at 2pm instead of 28th",
        "Can you reschedule our meeting to next week?",
        "I need to move the meeting to a different time",
        "Please update the booking to include more participants",
        "Let's modify the booking for tomorrow instead",
        "New date: let's meet on Friday instead"
    ]
    
    print("Testing modification keyword detection...")
    print("=" * 60)
    
    # Check if our keywords would catch the test cases
    all_detected = True
    
    for i, email_body in enumerate(test_emails, 1):
        detected = any(keyword.lower() in email_body.lower() for keyword in modification_phrases)
        status = "✅ DETECTED" if detected else "❌ MISSED"
        print(f"{i}. {status}: {email_body[:50]}...")
        if not detected:
            all_detected = False
    
    print("=" * 60)
    if all_detected:
        print("✅ SUCCESS: All modification language patterns detected!")
    else:
        print("❌ FAILURE: Some modification patterns missed")
        
    return all_detected

def test_enhanced_prompts():
    """Verify that our enhanced prompts include the modification keywords"""
    
    # Read the supervisor prompt section
    try:
        with open('/Users/samuelaudette/Documents/code_projects/agent_inbox_1.14/src/agents/supervisor.py', 'r') as f:
            supervisor_content = f.read()
    except FileNotFoundError:
        print("❌ Could not read supervisor.py file")
        return False
    
    # Read the email processor prompt section
    try:
        with open('/Users/samuelaudette/Documents/code_projects/agent_inbox_1.14/src/agents/email_processor.py', 'r') as f:
            processor_content = f.read()
    except FileNotFoundError:
        print("❌ Could not read email_processor.py file")
        return False
    
    print("Testing enhanced prompts...")
    print("=" * 60)
    
    # Check supervisor has modification detection
    supervisor_has_detection = (
        "CALENDAR MODIFICATION DETECTION" in supervisor_content and
        "change the event" in supervisor_content and
        "reschedule" in supervisor_content
    )
    
    # Check email processor has modification guidance
    processor_has_guidance = (
        "calendar modification language" in processor_content and
        "change the event/meeting" in processor_content and
        "modification" in processor_content
    )
    
    print(f"Supervisor modification detection: {'✅ FOUND' if supervisor_has_detection else '❌ MISSING'}")
    print(f"Email processor guidance: {'✅ FOUND' if processor_has_guidance else '❌ MISSING'}")
    
    print("=" * 60)
    success = supervisor_has_detection and processor_has_guidance
    if success:
        print("✅ SUCCESS: Enhanced prompts contain modification detection logic!")
    else:
        print("❌ FAILURE: Enhanced prompts missing modification detection")
        
    return success

if __name__ == "__main__":
    print("Calendar Modification Fix Validation")
    print("=" * 60)
    
    test1_result = test_modification_keywords()
    print()
    test2_result = test_enhanced_prompts()
    
    print("\n" + "=" * 60)
    if test1_result and test2_result:
        print("🎉 OVERALL SUCCESS: Calendar modification detection improvements validated!")
        print("\nNext steps:")
        print("- Test with real email workflow using CLI")
        print("- Monitor LangSmith traces for improved routing decisions")
    else:
        print("❌ OVERALL FAILURE: Some validation checks failed")
    print("=" * 60)
