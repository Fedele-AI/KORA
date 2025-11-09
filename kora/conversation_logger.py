"""
Conversation Logger for KORA
Logs user messages and AI responses to SQLite/PostgreSQL database
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from prisma import Prisma
from prisma.models import Session, Message, IngestionLog

# Initialize Prisma client
_prisma_client: Optional[Prisma] = None
_db_enabled = os.getenv("ENABLE_DB_LOGGING", "true").lower() == "true"


async def get_db() -> Prisma:
    """Get or create Prisma database connection"""
    global _prisma_client
    
    if not _db_enabled:
        raise RuntimeError("Database logging is disabled")
    
    if _prisma_client is None:
        _prisma_client = Prisma()
        await _prisma_client.connect()
    
    return _prisma_client


async def close_db():
    """Close database connection"""
    global _prisma_client
    
    if _prisma_client is not None:
        await _prisma_client.disconnect()
        _prisma_client = None


@asynccontextmanager
async def db_context():
    """Context manager for database operations"""
    try:
        db = await get_db()
        yield db
    except Exception as e:
        print(f"Database error: {e}")
        raise


class ConversationLogger:
    """Logger for user-AI conversations"""
    
    def __init__(self):
        self.enabled = _db_enabled
        self.current_session_id: Optional[str] = None
    
    async def start_session(
        self,
        user_id: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Start a new conversation session"""
        if not self.enabled:
            return None
        
        try:
            async with db_context() as db:
                session = await db.session.create(
                    data={
                        "userId": user_id,
                        "model": model,
                        "metadata": json.dumps(metadata) if metadata else None,
                    }
                )
                self.current_session_id = session.id
                return session.id
        except Exception as e:
            print(f"Error starting session: {e}")
            return None
    
    async def end_session(self, session_id: Optional[str] = None):
        """End a conversation session"""
        if not self.enabled:
            return
        
        sid = session_id or self.current_session_id
        if not sid:
            return
        
        try:
            async with db_context() as db:
                await db.session.update(
                    where={"id": sid},
                    data={"endTime": datetime.now()}
                )
                
                if sid == self.current_session_id:
                    self.current_session_id = None
        except Exception as e:
            print(f"Error ending session: {e}")
    
    async def log_message(
        self,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Log a message (user or assistant)
        
        Args:
            role: "user" or "assistant"
            content: The message text
            session_id: Session ID (uses current session if not provided)
            model: AI model used
            tokens_used: Number of tokens
            latency_ms: Response time in milliseconds
            metadata: Additional context
        
        Returns:
            Message ID if successful, None otherwise
        """
        if not self.enabled:
            return None
        
        sid = session_id or self.current_session_id
        
        # If no session exists, create one
        if not sid:
            sid = await self.start_session()
        
        if not sid:
            return None
        
        try:
            async with db_context() as db:
                message = await db.message.create(
                    data={
                        "sessionId": sid,
                        "role": role,
                        "content": content,
                        "model": model,
                        "tokensUsed": tokens_used,
                        "latencyMs": latency_ms,
                        "metadata": json.dumps(metadata) if metadata else None,
                    }
                )
                
                # Auto-generate session title from first user message
                if role == "user":
                    session = await db.session.find_unique(where={"id": sid})
                    if session and not session.title:
                        # Use first 50 chars of user message as title
                        title = content[:50] + ("..." if len(content) > 50 else "")
                        await db.session.update(
                            where={"id": sid},
                            data={"title": title}
                        )
                
                return message.id
        except Exception as e:
            print(f"Error logging message: {e}")
            return None
    
    async def log_user_message(self, content: str, **kwargs) -> Optional[str]:
        """Convenience method for logging user messages"""
        return await self.log_message("user", content, **kwargs)
    
    async def log_assistant_message(self, content: str, **kwargs) -> Optional[str]:
        """Convenience method for logging assistant messages"""
        return await self.log_message("assistant", content, **kwargs)
    
    async def get_session_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all messages in a session"""
        if not self.enabled:
            return []
        
        try:
            async with db_context() as db:
                messages = await db.message.find_many(
                    where={"sessionId": session_id},
                    order={"timestamp": "asc"},
                    take=limit
                )
                
                return [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "model": msg.model,
                        "tokensUsed": msg.tokensUsed,
                        "latencyMs": msg.latencyMs,
                        "metadata": json.loads(msg.metadata) if msg.metadata else None,
                    }
                    for msg in messages
                ]
        except Exception as e:
            print(f"Error getting session history: {e}")
            return []
    
    async def get_recent_sessions(
        self,
        user_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent conversation sessions"""
        if not self.enabled:
            return []
        
        try:
            async with db_context() as db:
                where = {"userId": user_id} if user_id else {}
                
                sessions = await db.session.find_many(
                    where=where,
                    order={"startTime": "desc"},
                    take=limit,
                    include={"messages": {"take": 1, "order": {"timestamp": "asc"}}}
                )
                
                return [
                    {
                        "id": sess.id,
                        "userId": sess.userId,
                        "startTime": sess.startTime.isoformat(),
                        "endTime": sess.endTime.isoformat() if sess.endTime else None,
                        "title": sess.title,
                        "model": sess.model,
                        "messageCount": len(sess.messages),
                    }
                    for sess in sessions
                ]
        except Exception as e:
            print(f"Error getting recent sessions: {e}")
            return []
    
    async def log_ingestion(
        self,
        file_name: str,
        file_size: Optional[int] = None,
        file_type: Optional[str] = None,
        chunks_count: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> Optional[str]:
        """Log document ingestion"""
        if not self.enabled:
            return None
        
        try:
            async with db_context() as db:
                log = await db.ingestionlog.create(
                    data={
                        "fileName": file_name,
                        "fileSize": file_size,
                        "fileType": file_type,
                        "chunksCount": chunks_count,
                        "success": success,
                        "errorMessage": error_message,
                        "durationMs": duration_ms,
                        "userId": user_id,
                    }
                )
                return log.id
        except Exception as e:
            print(f"Error logging ingestion: {e}")
            return None


# Global conversation logger instance
conversation_logger = ConversationLogger()
