"""
Database logging module for KORA
Uses Prisma with SQLite (local) or PostgreSQL (production)
"""
import os
import json
import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from prisma import Prisma
from prisma.errors import PrismaError

logger = logging.getLogger(__name__)

# Global Prisma client instance
_db_client: Optional[Prisma] = None


def get_database_url() -> str:
    """Get database URL from environment or use default SQLite"""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        # Default to SQLite in .kora directory
        db_dir = os.path.expanduser("~/.kora")
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "kora.db")
        db_url = f"file:{db_path}"
    return db_url


async def get_db() -> Prisma:
    """Get or create database client"""
    global _db_client
    
    if _db_client is None:
        _db_client = Prisma(auto_register=True)
        try:
            await _db_client.connect()
            logger.info(f"Connected to database: {get_database_url()}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            _db_client = None
            raise
    
    return _db_client


async def close_db():
    """Close database connection"""
    global _db_client
    if _db_client is not None:
        await _db_client.disconnect()
        _db_client = None
        logger.info("Database connection closed")


@asynccontextmanager
async def db_context():
    """Context manager for database operations"""
    try:
        db = await get_db()
        yield db
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise


class DatabaseLogger:
    """Main database logger class"""
    
    def __init__(self):
        self.enabled = os.getenv("ENABLE_DB_LOGGING", "true").lower() == "true"
    
    async def log_activity(
        self,
        action: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        duration: Optional[float] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log user activity"""
        if not self.enabled:
            return
        
        try:
            async with db_context() as db:
                await db.activitylog.create(
                    data={
                        "userId": user_id,
                        "username": username,
                        "action": action,
                        "endpoint": endpoint,
                        "method": method,
                        "statusCode": status_code,
                        "duration": duration,
                        "ipAddress": ip_address,
                        "userAgent": user_agent,
                        "metadata": json.dumps(metadata) if metadata else None
                    }
                )
        except Exception as e:
            logger.error(f"Failed to log activity: {e}")
    
    async def log_query(
        self,
        query: str,
        response: str,
        model: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        documents_used: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        duration: Optional[float] = None,
        ip_address: Optional[str] = None
    ):
        """Log RAG query"""
        if not self.enabled:
            return
        
        try:
            async with db_context() as db:
                await db.querylog.create(
                    data={
                        "userId": user_id,
                        "username": username,
                        "query": query,
                        "response": response,
                        "model": model,
                        "temperature": temperature,
                        "topK": top_k,
                        "documentsUsed": documents_used,
                        "success": success,
                        "errorMessage": error_message,
                        "duration": duration,
                        "ipAddress": ip_address
                    }
                )
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
    
    async def log_auth(
        self,
        action: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        success: bool = True,
        reason: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Log authentication event"""
        if not self.enabled:
            return
        
        try:
            async with db_context() as db:
                await db.authlog.create(
                    data={
                        "action": action,
                        "userId": user_id,
                        "username": username,
                        "success": success,
                        "reason": reason,
                        "ipAddress": ip_address,
                        "userAgent": user_agent
                    }
                )
        except Exception as e:
            logger.error(f"Failed to log auth: {e}")
    
    async def log_system(
        self,
        level: str,
        component: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        exception: Optional[str] = None
    ):
        """Log system event"""
        if not self.enabled:
            return
        
        try:
            async with db_context() as db:
                await db.systemlog.create(
                    data={
                        "level": level.upper(),
                        "component": component,
                        "message": message,
                        "details": json.dumps(details) if details else None,
                        "exception": exception
                    }
                )
        except Exception as e:
            logger.error(f"Failed to log system event: {e}")
    
    async def log_api_usage(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        user_id: Optional[str] = None,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None
    ):
        """Log API usage"""
        if not self.enabled:
            return
        
        try:
            async with db_context() as db:
                await db.apiusage.create(
                    data={
                        "userId": user_id,
                        "endpoint": endpoint,
                        "method": method,
                        "statusCode": status_code,
                        "duration": duration,
                        "requestSize": request_size,
                        "responseSize": response_size
                    }
                )
        except Exception as e:
            logger.error(f"Failed to log API usage: {e}")
    
    async def log_ingestion(
        self,
        file_name: str,
        file_size: Optional[int] = None,
        file_type: Optional[str] = None,
        chunks_count: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        duration: Optional[float] = None,
        user_id: Optional[str] = None
    ):
        """Log document ingestion"""
        if not self.enabled:
            return
        
        try:
            async with db_context() as db:
                await db.ingestionlog.create(
                    data={
                        "fileName": file_name,
                        "fileSize": file_size,
                        "fileType": file_type,
                        "chunksCount": chunks_count,
                        "success": success,
                        "errorMessage": error_message,
                        "duration": duration,
                        "userId": user_id
                    }
                )
        except Exception as e:
            logger.error(f"Failed to log ingestion: {e}")
    
    async def get_query_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ):
        """Get query history"""
        try:
            async with db_context() as db:
                where_clause = {"userId": user_id} if user_id else {}
                logs = await db.querylog.find_many(
                    where=where_clause,
                    order={"timestamp": "desc"},
                    take=limit,
                    skip=offset
                )
                return logs
        except Exception as e:
            logger.error(f"Failed to get query history: {e}")
            return []
    
    async def get_stats(self):
        """Get usage statistics"""
        try:
            async with db_context() as db:
                total_queries = await db.querylog.count()
                total_users = await db.activitylog.group_by(
                    by=["userId"],
                    count=True
                )
                
                return {
                    "total_queries": total_queries,
                    "unique_users": len(total_users),
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


# Global logger instance
db_logger = DatabaseLogger()


# Decorator for logging function calls
def log_function_call(action: str, component: str):
    """Decorator to log function calls"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_msg = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                duration = time.time() - start_time
                try:
                    await db_logger.log_system(
                        level="INFO" if success else "ERROR",
                        component=component,
                        message=f"{action}: {func.__name__}",
                        details={
                            "function": func.__name__,
                            "duration": duration,
                            "success": success,
                            "error": error_msg
                        }
                    )
                except:
                    pass  # Don't let logging errors break the application
        
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, just call them
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
