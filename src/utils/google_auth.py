"""
Google Authentication Helper
Handles OAuth2 authentication for Google Workspace APIs
"""

import os
import pickle
from typing import Optional, List
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import json


class GoogleAuthHelper:
    """Helper class for Google API authentication"""
    
    @staticmethod
    def get_credentials(scopes: List[str], token_file: str) -> Optional[Credentials]:
        """
        Get or create credentials for Google APIs
        
        Args:
            scopes: List of API scopes required
            token_file: Path to store the token pickle file
            
        Returns:
            Credentials object or None if authentication fails
        """
        creds = None
        
        # Try to load existing token
        if os.path.exists(token_file):
            try:
                with open(token_file, 'rb') as token:
                    creds = pickle.load(token)
            except Exception as e:
                print(f"Error loading token from {token_file}: {e}")
        
        # If there are no (valid) credentials available, try to create from env
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Error refreshing token: {e}")
                    creds = None
            
            # Try to create from environment variables
            if not creds:
                client_id = os.getenv('GOOGLE_CLIENT_ID')
                client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
                refresh_token = os.getenv('GMAIL_REFRESH_TOKEN')
                
                if all([client_id, client_secret, refresh_token]):
                    try:
                        # Create credentials from refresh token
                        creds = Credentials(
                            None,
                            refresh_token=refresh_token,
                            token_uri="https://oauth2.googleapis.com/token",
                            client_id=client_id,
                            client_secret=client_secret,
                            scopes=scopes
                        )
                        creds.refresh(Request())
                        
                        # Save the credentials for next run
                        with open(token_file, 'wb') as token:
                            pickle.dump(creds, token)
                            
                    except Exception as e:
                        print(f"Error creating credentials from refresh token: {e}")
                        print("The refresh token may not have the required scopes.")
                        creds = None
                
                # If env vars don't work, try credentials.json file
                if not creds and os.path.exists('credentials.json'):
                    try:
                        flow = InstalledAppFlow.from_client_secrets_file(
                            'credentials.json', scopes)
                        creds = flow.run_local_server(port=0)
                        
                        # Save the credentials
                        with open(token_file, 'wb') as token:
                            pickle.dump(creds, token)
                            
                    except Exception as e:
                        print(f"Error with OAuth flow: {e}")
                        creds = None
        
        return creds
    
    @staticmethod
    def create_mock_service(service_name: str, version: str):
        """
        Create a mock service for testing when credentials are not available
        """
        class MockService:
            def __init__(self, name):
                self.name = name
                print(f"Warning: Using mock {name} service - API calls will not work")
            
            def __getattr__(self, name):
                def mock_method(*args, **kwargs):
                    print(f"Mock call to {self.name}.{name}")
                    return self
                return mock_method
            
            def execute(self):
                return {"mock": True, "message": "This is a mock response"}
        
        return MockService(f"{service_name}_{version}")
