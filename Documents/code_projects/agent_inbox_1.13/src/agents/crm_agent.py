"""
CRM Agent (Customer Relationship Management)
Handles contact information and task delegation using Google Contacts API
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import os

from googleapiclient.errors import HttpError
from src.utils.google_auth import GoogleAuthHelper

from langsmith import traceable
from src.agents.base_agent import BaseAgent
from src.models.state import AgentState, ContactData


class CRMAgent(BaseAgent):
    """
    Agent responsible for CRM operations:
    - Look up contact information
    - Track relationships and context
    - Handle task delegation
    - Manage contact-related requests
    """
    
    SCOPES = ['https://www.googleapis.com/auth/contacts.readonly']
    
    def __init__(self):
        super().__init__(
            name="crm_agent",
            model="gpt-4o",
            temperature=0.1
        )
        self.service = None
        self._initialize_people_service()
    
    def _initialize_people_service(self):
        """Initialize Google People (Contacts) service with OAuth2"""
        try:
            creds = GoogleAuthHelper.get_credentials(self.SCOPES, 'token_contacts.pickle')
            if creds:
                from googleapiclient.discovery import build
                self.service = build('people', 'v1', credentials=creds)
                self.logger.info("Google People service initialized")
            else:
                self.logger.warning("Failed to get Google People credentials, using mock service")
                self.service = GoogleAuthHelper.create_mock_service('people', 'v1')
        except Exception as e:
            self.logger.error(f"Error initializing People service: {e}")
            self.service = GoogleAuthHelper.create_mock_service('people', 'v1')
    
    @traceable(name="crm_process", tags=["agent", "crm"])
    async def process(self, state: AgentState) -> AgentState:
        """
        Process CRM-related requests and task delegation
        """
        try:
            self.logger.info("Processing CRM/task delegation request")
            
            # Extract contact queries and task delegation details
            crm_request = await self._extract_crm_request(state)
            
            # Initialize contact data if not exists
            if not state.contact_data:
                state.contact_data = ContactData()
            
            # Look up contacts mentioned in the email
            contacts = []
            unknown_contacts = []
            
            for contact_query in crm_request.get("contact_queries", []):
                found_contacts = await self._search_contacts(contact_query)
                
                if found_contacts:
                    # Enrich contact information
                    for contact in found_contacts:
                        enriched = await self._enrich_contact_info(contact)
                        contacts.append(enriched)
                else:
                    unknown_contacts.append(contact_query)
            
            state.contact_data.contacts = contacts
            state.contact_data.unknown_contacts = unknown_contacts
            
            # Process task delegation if needed
            if crm_request.get("is_task_delegation"):
                delegation_context = await self._process_task_delegation(
                    crm_request,
                    contacts
                )
                state.contact_data.relationship_context = delegation_context
            
            # Generate CRM summary
            crm_summary = self._generate_crm_summary(state.contact_data, crm_request)
            
            # Add summary message
            self._add_message(
                state,
                crm_summary,
                metadata={
                    "contact_data": state.contact_data.dict(),
                    "contacts_found": len(contacts),
                    "task_delegation": crm_request.get("is_task_delegation", False)
                }
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"CRM agent failed: {str(e)}")
            state.add_error(f"CRM processing failed: {str(e)}")
            return state
    
    async def _extract_crm_request(self, state: AgentState) -> Dict[str, Any]:
        """Extract CRM-related information from email using LLM"""
        prompt = f"""Analyze this email for CRM and task delegation needs:

Subject: {state.email.subject}
From: {state.email.sender}
Body: {state.email.body}

Extracted Context:
- Key entities: {state.extracted_context.key_entities if state.extracted_context else 'None'}
- Requested actions: {state.extracted_context.requested_actions if state.extracted_context else 'None'}

Return JSON with:
{{
    "contact_queries": ["list of people/companies to look up"],
    "is_task_delegation": true/false,
    "delegation_details": {{
        "task_description": "what needs to be done",
        "assignees": ["who should do it"],
        "deadline": "when it needs to be done",
        "priority": "high/medium/low"
    }},
    "relationship_context_needed": true/false,
    "contact_info_needed": ["email", "phone", "organization", "role"]
}}"""

        response = await self._call_llm(prompt)
        return json.loads(response)
    
    async def _search_contacts(self, query: str) -> List[Dict[str, Any]]:
        """Search Google Contacts for people matching the query"""
        try:
            # Search for contacts
            results = self.service.people().searchContacts(
                query=query,
                readMask='names,emailAddresses,phoneNumbers,organizations,biographies'
            ).execute()
            
            contacts = []
            for person in results.get('results', []):
                contact_data = person.get('person', {})
                
                # Extract contact information
                contact = {
                    'resourceName': contact_data.get('resourceName'),
                    'name': self._get_display_name(contact_data.get('names', [])),
                    'emails': [
                        {'address': e.get('value'), 'type': e.get('type', 'other')}
                        for e in contact_data.get('emailAddresses', [])
                    ],
                    'phones': [
                        {'number': p.get('value'), 'type': p.get('type', 'other')}
                        for p in contact_data.get('phoneNumbers', [])
                    ],
                    'organizations': [
                        {
                            'name': o.get('name'),
                            'title': o.get('title'),
                            'department': o.get('department')
                        }
                        for o in contact_data.get('organizations', [])
                    ],
                    'notes': contact_data.get('biographies', [{}])[0].get('value', '') if contact_data.get('biographies') else '',
                    'search_query': query
                }
                
                contacts.append(contact)
            
            return contacts
            
        except HttpError as error:
            self.logger.error(f"People API error: {error}")
            return []
    
    async def _enrich_contact_info(self, contact: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich contact information with additional context"""
        # For now, just add some computed fields
        # In a real implementation, this could query additional systems
        
        contact['primary_email'] = contact['emails'][0]['address'] if contact['emails'] else None
        contact['primary_phone'] = contact['phones'][0]['number'] if contact['phones'] else None
        contact['current_organization'] = contact['organizations'][0]['name'] if contact['organizations'] else None
        contact['current_title'] = contact['organizations'][0]['title'] if contact['organizations'] else None
        
        # Add a summary
        summary_parts = []
        if contact['current_title'] and contact['current_organization']:
            summary_parts.append(f"{contact['current_title']} at {contact['current_organization']}")
        if contact['primary_email']:
            summary_parts.append(contact['primary_email'])
        
        contact['summary'] = " | ".join(summary_parts)
        
        return contact
    
    async def _process_task_delegation(
        self,
        crm_request: Dict[str, Any],
        contacts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process task delegation details and prepare context"""
        delegation = crm_request.get("delegation_details", {})
        
        # Match assignees with contacts
        assignee_contacts = []
        for assignee in delegation.get("assignees", []):
            # Find matching contact
            for contact in contacts:
                if (assignee.lower() in contact['name'].lower() or 
                    assignee.lower() in contact.get('primary_email', '').lower()):
                    assignee_contacts.append({
                        'name': contact['name'],
                        'email': contact.get('primary_email'),
                        'title': contact.get('current_title'),
                        'organization': contact.get('current_organization')
                    })
                    break
        
        # Build delegation context
        context = {
            'task': delegation.get('task_description'),
            'assignees': assignee_contacts,
            'deadline': delegation.get('deadline'),
            'priority': delegation.get('priority'),
            'delegation_ready': len(assignee_contacts) > 0,
            'missing_assignees': [
                a for a in delegation.get("assignees", [])
                if not any(a.lower() in str(ac).lower() for ac in assignee_contacts)
            ]
        }
        
        return context
    
    def _generate_crm_summary(self, contact_data: ContactData, crm_request: Dict[str, Any]) -> str:
        """Generate a summary of CRM processing results"""
        summary_parts = []
        
        # Contact lookup summary
        if contact_data.contacts:
            summary_parts.append(f"👥 Found {len(contact_data.contacts)} contacts:")
            for contact in contact_data.contacts[:3]:  # Show top 3
                summary_parts.append(f"  • {contact['name']}: {contact.get('summary', 'No details')}")
        
        if contact_data.unknown_contacts:
            summary_parts.append(f"❓ Could not find: {', '.join(contact_data.unknown_contacts)}")
        
        # Task delegation summary
        if crm_request.get("is_task_delegation") and contact_data.relationship_context:
            ctx = contact_data.relationship_context
            summary_parts.append(f"\n📋 Task Delegation:")
            summary_parts.append(f"  Task: {ctx.get('task')}")
            summary_parts.append(f"  Priority: {ctx.get('priority')}")
            if ctx.get('deadline'):
                summary_parts.append(f"  Deadline: {ctx.get('deadline')}")
            
            if ctx.get('delegation_ready'):
                summary_parts.append(f"  ✓ Ready to delegate to {len(ctx.get('assignees', []))} people")
            else:
                summary_parts.append(f"  ⚠️ Missing contact info for some assignees")
        
        return "\n".join(summary_parts) if summary_parts else "CRM processing completed"
    
    def _get_display_name(self, names: List[Dict[str, Any]]) -> str:
        """Extract the best display name from Google Contacts name data"""
        if not names:
            return "Unknown"
        
        # Prefer the primary name
        for name in names:
            if name.get('metadata', {}).get('primary'):
                return name.get('displayName', 'Unknown')
        
        # Fall back to first name
        return names[0].get('displayName', 'Unknown')
