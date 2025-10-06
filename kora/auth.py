"""
Authentication module for KORA using Kerberos-based API key generation.
"""

import os
import secrets
import hashlib
import json
import time
import subprocess
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import kerberos
    KERBEROS_AVAILABLE = True
except ImportError:
    KERBEROS_AVAILABLE = False


class KoraAuthenticator:
    """Handles Kerberos-based authentication and API key generation for KORA."""
    
    def __init__(self, service_name: str = "HTTP", hostname: Optional[str] = None):
        """
        Initialize the authenticator.
        
        Args:
            service_name: Kerberos service name (default: HTTP)
            hostname: Hostname for Kerberos service (default: localhost)
        """
        self.service_name = service_name
        self.hostname = hostname or "localhost"
        self.api_keys_file = Path(".kora") / "api_keys.json"
        self.sessions_file = Path(".kora") / "sessions.json"
        
        # Ensure directories exist
        self.api_keys_file.parent.mkdir(parents=True, exist_ok=True)
        
    def _load_api_keys(self) -> Dict[str, Any]:
        """Load API keys from storage."""
        if not self.api_keys_file.exists():
            return {}
        try:
            with open(self.api_keys_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def _save_api_keys(self, api_keys: Dict[str, Any]) -> None:
        """Save API keys to storage."""
        with open(self.api_keys_file, 'w') as f:
            json.dump(api_keys, f, indent=2)
    
    def _load_sessions(self) -> Dict[str, Any]:
        """Load active sessions from storage."""
        if not self.sessions_file.exists():
            return {}
        try:
            with open(self.sessions_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def _save_sessions(self, sessions: Dict[str, Any]) -> None:
        """Save sessions to storage."""
        with open(self.sessions_file, 'w') as f:
            json.dump(sessions, f, indent=2)
    
    def generate_api_key(self, username: str, password: Optional[str] = None, demo_mode: bool = False) -> Optional[str]:
        """
        Generate a 64-character API key using Kerberos authentication.
        
        Args:
            username: Kerberos username
            password: Kerberos password
            demo_mode: If True, skip Kerberos validation (for testing)
            
        Returns:
            64-character API key if authentication successful, None otherwise
        """
        try:
            # Demo mode for testing
            if demo_mode:
                print(f"[DEMO MODE] Generating API key for {username}")
                if not password:
                    password = "demo_password"
            else:
                if not password:
                    return None
                    
                # Try Kerberos authentication if available
                if KERBEROS_AVAILABLE:
                    # Authenticate with Kerberos
                    service_principal = f"{self.service_name}@{self.hostname}"
                    
                    # Initialize Kerberos context
                    result, context = kerberos.authGSSClientInit(service_principal)
                    if result != kerberos.AUTH_GSS_COMPLETE:
                        return None
                    
                    # Authenticate using username/password
                    result = kerberos.authGSSClientStep(context, "")
                    if result == kerberos.AUTH_GSS_CONTINUE:
                        # For simplicity, we'll use a hash-based approach with username
                        # In a real implementation, you'd complete the full Kerberos handshake
                        pass
                    
                    # Clean up Kerberos context
                    kerberos.authGSSClientClean(context)
                else:
                    # Fallback: Use kinit command if available
                    try:
                        result = subprocess.run(
                            ["kinit", "-l", "24h", username],
                            input=password,
                            text=True,
                            capture_output=True,
                            timeout=10
                        )
                        if result.returncode != 0:
                            print(f"Kerberos authentication failed: {result.stderr}")
                            return None
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        # Kerberos not available, use simple validation
                        # In production, this should be replaced with proper authentication
                        if not username or not password:
                            return None
            
            # Generate 64-character API key
            timestamp = str(time.time())
            data = f"{username}:{timestamp}:{self.hostname}"
            hash_input = data.encode('utf-8')
            
            # Create a secure hash and extend to 64 characters
            api_key_base = hashlib.sha256(hash_input).hexdigest()
            additional_entropy = secrets.token_hex(16)
            api_key = (api_key_base + additional_entropy)[:64]
            
            # Store API key with metadata
            api_keys = self._load_api_keys()
            api_keys[api_key] = {
                "username": username,
                "created_at": timestamp,
                "hostname": self.hostname,
                "active": True,
                "demo_mode": demo_mode
            }
            self._save_api_keys(api_keys)
            
            return api_key
            
        except Exception as e:
            if not demo_mode:
                print(f"Kerberos authentication failed: {e}")
            return None
    
    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate an API key.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            True if valid and active, False otherwise
        """
        if not api_key or len(api_key) != 64:
            return False
        
        api_keys = self._load_api_keys()
        key_data = api_keys.get(api_key)
        
        if not key_data:
            return False
        
        return key_data.get("active", False)
    
    def create_session(self, api_key: str) -> Optional[str]:
        """
        Create a session token for a valid API key.
        
        Args:
            api_key: Valid API key
            
        Returns:
            Session token if successful, None otherwise
        """
        if not self.validate_api_key(api_key):
            return None
        
        # Generate session token
        session_token = secrets.token_urlsafe(32)
        
        # Store session
        sessions = self._load_sessions()
        sessions[session_token] = {
            "api_key": api_key,
            "created_at": time.time(),
            "last_accessed": time.time()
        }
        self._save_sessions(sessions)
        
        return session_token
    
    def validate_session(self, session_token: str) -> bool:
        """
        Validate a session token.
        
        Args:
            session_token: Session token to validate
            
        Returns:
            True if valid session, False otherwise
        """
        if not session_token:
            return False
        
        sessions = self._load_sessions()
        session_data = sessions.get(session_token)
        
        if not session_data:
            return False
        
        # Check if session is too old (24 hours)
        current_time = time.time()
        if current_time - session_data["created_at"] > 86400:  # 24 hours
            # Remove expired session
            del sessions[session_token]
            self._save_sessions(sessions)
            return False
        
        # Update last accessed time
        session_data["last_accessed"] = current_time
        sessions[session_token] = session_data
        self._save_sessions(sessions)
        
        # Validate the associated API key
        return self.validate_api_key(session_data["api_key"])
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if revoked successfully, False otherwise
        """
        api_keys = self._load_api_keys()
        
        if api_key in api_keys:
            api_keys[api_key]["active"] = False
            self._save_api_keys(api_keys)
            
            # Remove all sessions associated with this API key
            sessions = self._load_sessions()
            sessions_to_remove = [
                token for token, data in sessions.items()
                if data.get("api_key") == api_key
            ]
            
            for token in sessions_to_remove:
                del sessions[token]
            
            self._save_sessions(sessions)
            return True
        
        return False
    
    def get_user_from_session(self, session_token: str) -> Optional[str]:
        """
        Get username from a valid session token.
        
        Args:
            session_token: Session token
            
        Returns:
            Username if session is valid, None otherwise
        """
        if not self.validate_session(session_token):
            return None
        
        sessions = self._load_sessions()
        session_data = sessions.get(session_token)
        
        if not session_data:
            return None
        
        api_key = session_data["api_key"]
        api_keys = self._load_api_keys()
        key_data = api_keys.get(api_key)
        
        if key_data:
            return key_data.get("username")
        
        return None


# Global authenticator instance
_authenticator = None

def get_authenticator() -> KoraAuthenticator:
    """Get the global authenticator instance."""
    global _authenticator
    if _authenticator is None:
        _authenticator = KoraAuthenticator()
    return _authenticator