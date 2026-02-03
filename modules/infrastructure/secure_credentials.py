"""
Secure Credential Manager
Author: Erdinc Erdogan
Purpose: Manages API keys and session tokens securely via environment variables with expiration checking and credential masking for logging.
References:
- Secure Credential Storage Patterns
- Environment Variable Security
- Token Expiration Management
Usage:
    manager = SecureCredentialManager()
    credentials = manager.get_credentials()
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime, timezone

@dataclass
class CloudCredentials:
    """Secure cloud credentials container."""
    api_key: str
    session_token: Optional[str] = None
    expires_at: Optional[str] = None
    
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        exp = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) > exp
    
    def mask(self) -> str:
        """Return masked credential for logging."""
        if len(self.api_key) > 8:
            return f"{self.api_key[:4]}****{self.api_key[-4:]}"
        return "****"

class SecureCredentialManager:
    """Manages credentials securely via environment variables."""
    
    ENV_API_KEY = "CAMBER_API_KEY"
    ENV_SESSION_TOKEN = "CAMBER_SESSION_TOKEN"
    ENV_API_SECRET = "CAMBER_API_SECRET"
    
    def __init__(self):
        self._credentials: Optional[CloudCredentials] = None
        self._load_from_env()
    
    def _load_from_env(self):
        """Load credentials from environment variables."""
        api_key = os.environ.get(self.ENV_API_KEY)
        session_token = os.environ.get(self.ENV_SESSION_TOKEN)
        
        if api_key:
            self._credentials = CloudCredentials(
                api_key=api_key,
                session_token=session_token
            )
    
    def get_credentials(self) -> Optional[CloudCredentials]:
        """Get current credentials (never expose raw key in logs)."""
        return self._credentials
    
    def is_authenticated(self) -> bool:
        """Check if valid credentials exist."""
        if not self._credentials:
            return False
        return not self._credentials.is_expired()
    
    def get_masked_key(self) -> str:
        """Get masked API key for logging."""
        if self._credentials:
            return self._credentials.mask()
        return "NOT_SET"
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        if not self._credentials:
            return {}
        
        headers = {"Authorization": f"Bearer {self._credentials.api_key}"}
        if self._credentials.session_token:
            headers["X-Session-Token"] = self._credentials.session_token
        return headers
    
    def validate_no_hardcoded(self, code_content: str) -> bool:
        """Scan code for hardcoded credentials."""
        danger_patterns = ["api_key =", "API_KEY =", "sk-", "secret ="]
        for pattern in danger_patterns:
            if pattern in code_content and "environ" not in code_content:
                return False
        return True
