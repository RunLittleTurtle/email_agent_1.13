"""
RAG Agent (Retrieval-Augmented Generation)
Handles document requests and information retrieval using Google Drive API
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import io

from googleapiclient.errors import HttpError
from src.utils.google_auth import GoogleAuthHelper
from googleapiclient.http import MediaIoBaseDownload
import pickle

from langsmith import traceable
from langgraph.runtime import Runtime
from src.agents.base_agent import BaseAgent
from src.models.state import AgentState, DocumentData
from src.models.context import RuntimeContext


class RAGAgent(BaseAgent):
    """
    Agent responsible for document retrieval and search:
    - Search Google Drive for documents
    - Extract relevant information from documents
    - Provide document summaries
    - Handle information requests
    """
    
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    
    def __init__(self):
        super().__init__(
            name="rag_agent",
            model="gpt-4o",
            temperature=0.1
        )
        self.service = None
        self._initialize_drive_service()
    
    def _initialize_drive_service(self):
        """Initialize Google Drive service with OAuth2"""
        try:
            creds = GoogleAuthHelper.get_credentials(self.SCOPES, 'token_drive.pickle')
            if creds:
                from googleapiclient.discovery import build
                self.service = build('drive', 'v3', credentials=creds)
                self.logger.info("Google Drive service initialized")
            else:
                self.logger.warning("Failed to get Google Drive credentials, using mock service")
                self.service = GoogleAuthHelper.create_mock_service('drive', 'v3')
        except Exception as e:
            self.logger.error(f"Error initializing Drive service: {e}")
            self.service = GoogleAuthHelper.create_mock_service('drive', 'v3')
    
    @traceable(name="rag_process", tags=["agent", "rag"])
    async def process(self, state: AgentState, runtime: Optional[Runtime[RuntimeContext]] = None) -> Dict[str, Any]:
        """
        Process document requests and information retrieval
        """
        try:
            self.logger.info("Processing document/information request")
            
            # Extract search queries from email
            search_queries = await self._extract_search_queries(state)
            
            # Initialize document data if not exists
            document_data = state.document_data or DocumentData()
            document_data.search_queries = search_queries
            
            # Search for documents
            all_found_documents = []
            missing_queries = []
            
            for query in search_queries:
                documents = await self._search_documents(query)
                
                if documents:
                    # Get document content for top results
                    for doc in documents[:3]:  # Limit to top 3 per query
                        content = await self._get_document_content(doc)
                        if content:
                            doc['content_preview'] = content[:500]  # First 500 chars
                            doc['content_summary'] = await self._summarize_content(content, query)
                    
                    all_found_documents.extend(documents)
                else:
                    missing_queries.append(query)
            
            document_data.found_documents = all_found_documents
            document_data.missing_documents = missing_queries
            
            # Generate RAG summary with temporary state for response generation
            temp_state = state.copy()
            temp_state.document_data = document_data
            rag_summary = await self._generate_rag_response(temp_state)
            
            # Create AI message with RAG results
            rag_message = self.create_ai_message(
                rag_summary,
                metadata={
                    "document_data": document_data.dict(),
                    "documents_found": len(all_found_documents),
                    "queries_processed": len(search_queries)
                }
            )
            
            return {
                "messages": [rag_message],
                "document_data": document_data
            }
            
        except Exception as e:
            self.logger.error(f"RAG agent failed: {str(e)}")
            return {
                "error_messages": [f"Document retrieval failed: {str(e)}"]
            }
    
    async def _extract_search_queries(self, state: AgentState) -> List[str]:
        """Extract document search queries from email using LLM"""
        prompt = f"""Extract document search queries from this email:

Subject: {state.email.subject}
From: {state.email.sender}
Body: {state.email.body}

Extracted Context:
- Requested actions: {state.extracted_context.requested_actions if state.extracted_context else 'None'}
- Key entities: {state.extracted_context.key_entities if state.extracted_context else 'None'}

Identify what documents or information the sender is looking for.
Return JSON with a list of search queries:
{{
    "search_queries": [
        "query1",
        "query2"
    ],
    "document_types": ["presentation", "spreadsheet", "document", "pdf"],
    "context": "Brief context about what they're looking for"
}}"""

        response = await self._call_llm(prompt)
        result = json.loads(response)
        return result.get("search_queries", [])
    
    async def _search_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search Google Drive for documents matching the query"""
        try:
            # Build search query
            search_query = f"fullText contains '{query}' and trashed = false"
            
            # Search for files
            results = self.service.files().list(
                q=search_query,
                pageSize=10,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, owners, webViewLink)"
            ).execute()
            
            documents = []
            for file in results.get('files', []):
                documents.append({
                    'id': file['id'],
                    'name': file['name'],
                    'type': self._get_document_type(file['mimeType']),
                    'mime_type': file['mimeType'],
                    'modified': file.get('modifiedTime', ''),
                    'owner': file.get('owners', [{}])[0].get('displayName', 'Unknown'),
                    'link': file.get('webViewLink', ''),
                    'search_query': query
                })
            
            return documents
            
        except HttpError as error:
            self.logger.error(f"Drive API error: {error}")
            return []
    
    async def _get_document_content(self, document: Dict[str, Any]) -> Optional[str]:
        """Get content from a document (limited to Google Docs for now)"""
        try:
            mime_type = document['mime_type']
            
            # Handle Google Docs
            if mime_type == 'application/vnd.google-apps.document':
                # Export as plain text
                request = self.service.files().export_media(
                    fileId=document['id'],
                    mimeType='text/plain'
                )
                file = io.BytesIO()
                downloader = MediaIoBaseDownload(file, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                
                content = file.getvalue().decode('utf-8')
                return content
            
            # Handle text files
            elif mime_type.startswith('text/'):
                request = self.service.files().get_media(fileId=document['id'])
                file = io.BytesIO()
                downloader = MediaIoBaseDownload(file, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                
                content = file.getvalue().decode('utf-8')
                return content
            
            # For other types, return None (could be extended)
            return None
            
        except HttpError as error:
            self.logger.error(f"Error getting document content: {error}")
            return None
    
    async def _summarize_content(self, content: str, query: str) -> str:
        """Summarize document content relevant to the query"""
        prompt = f"""Summarize this document content focusing on information relevant to the query.

Query: {query}

Document content (truncated):
{content[:2000]}

Provide a brief summary (2-3 sentences) of relevant information found."""

        return await self._call_llm(prompt)
    
    async def _generate_rag_response(self, state: AgentState) -> str:
        """Generate a comprehensive RAG response based on found documents"""
        if not state.document_data.found_documents:
            return "📄 No documents found matching the requested information."
        
        # Build document context
        doc_context = []
        for doc in state.document_data.found_documents[:5]:  # Top 5 documents
            doc_info = f"- {doc['name']} ({doc['type']})"
            if doc.get('content_summary'):
                doc_info += f": {doc['content_summary']}"
            doc_context.append(doc_info)
        
        prompt = f"""Based on the document search results, provide a helpful response to the email request.

Original request: {state.email.body if state.email else 'Unknown'}

Found documents:
{chr(10).join(doc_context)}

Missing information for queries: {', '.join(state.document_data.missing_documents)}

Generate a response that:
1. Summarizes what documents were found
2. Highlights key information from the documents
3. Notes any information that couldn't be found
4. Suggests next steps if needed"""

        rag_response = await self._call_llm(prompt)
        
        return f"📄 Document Search Results:\n\n{rag_response}"
    
    def _get_document_type(self, mime_type: str) -> str:
        """Map MIME type to readable document type"""
        type_map = {
            'application/vnd.google-apps.document': 'Google Doc',
            'application/vnd.google-apps.spreadsheet': 'Google Sheet',
            'application/vnd.google-apps.presentation': 'Google Slides',
            'application/pdf': 'PDF',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'Word',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'Excel',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'PowerPoint',
            'text/plain': 'Text',
            'text/csv': 'CSV'
        }
        
        return type_map.get(mime_type, 'File')
