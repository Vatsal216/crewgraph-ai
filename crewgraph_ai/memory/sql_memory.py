"""
SQL Memory Backend for CrewGraph AI
Production-ready SQL database storage with multiple database support

Author: Vatsal216
Created: 2025-07-22 12:05:13 UTC
"""

import time
import json
import pickle
import gzip
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

try:
    from sqlalchemy import (
        create_engine, Column, String, Text, DateTime, Integer, 
        LargeBinary, Boolean, Index, text
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import QueuePool
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
    # SQLAlchemy Base
    Base = declarative_base()
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    # Create dummy base for when SQLAlchemy is not available
    class Base:
        pass

from .base import BaseMemory, MemoryOperation
from ..utils.logging import get_logger
from ..utils.exceptions import MemoryError

logger = get_logger(__name__)

# Define SQLAlchemy models only if SQLAlchemy is available
if SQLALCHEMY_AVAILABLE:
    class MemoryItem(Base):
        """SQLAlchemy model for memory items"""
        __tablename__ = 'crewgraph_memory'
        
        key = Column(String(255), primary_key=True)
        value_data = Column(LargeBinary)  # Pickled/compressed data
        value_type = Column(String(50))   # Type information
        metadata_json = Column(Text)      # JSON metadata
        created_at = Column(DateTime, default=datetime.utcnow)
        expires_at = Column(DateTime, nullable=True)
        access_count = Column(Integer, default=0)
        last_access = Column(DateTime, nullable=True)
        size_bytes = Column(Integer, default=0)
        is_compressed = Column(Boolean, default=False)
        
        # Indexes for performance
        __table_args__ = (
            Index('idx_expires_at', 'expires_at'),
            Index('idx_created_at', 'created_at'),
            Index('idx_last_access', 'last_access'),
        )
else:
    # Dummy class when SQLAlchemy is not available
    class MemoryItem:
        pass


class SQLMemory(BaseMemory):
    """
    SQL-based memory backend for CrewGraph AI.
    
    Provides enterprise-grade persistent storage with support for:
    - Multiple database engines (PostgreSQL, MySQL, SQLite, SQL Server)
    - Connection pooling and transaction management
    - Automatic schema creation and migration
    - TTL management with background cleanup
    - Compression and serialization options
    - Full-text search capabilities
    - Backup and restore functionality
    - Performance monitoring and optimization
    
    Supported databases:
    - PostgreSQL (recommended for production)
    - MySQL/MariaDB
    - SQLite (for development/testing)
    - SQL Server
    - Oracle (with appropriate drivers)
    
    Created by: Vatsal216
    Date: 2025-07-22 12:05:13 UTC
    """
    
    def __init__(self, 
                 database_url: str,
                 table_prefix: str = "crewgraph_",
                 enable_compression: bool = True,
                 compression_threshold: int = 1024,
                 serialization_format: str = "pickle",
                 pool_size: int = 10,
                 max_overflow: int = 20,
                 pool_timeout: int = 30,
                 enable_ttl_cleanup: bool = True,
                 ttl_cleanup_interval: int = 3600):
        """
        Initialize SQL memory backend.
        
        Args:
            database_url: SQLAlchemy database URL
            table_prefix: Prefix for database tables
            enable_compression: Enable data compression
            compression_threshold: Minimum size for compression
            serialization_format: Serialization format ('pickle', 'json')
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            pool_timeout: Connection timeout
            enable_ttl_cleanup: Enable automatic TTL cleanup
            ttl_cleanup_interval: TTL cleanup interval in seconds
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is not installed. Install with: pip install sqlalchemy")
        
        super().__init__()
        
        self.database_url = database_url
        self.table_prefix = table_prefix
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.serialization_format = serialization_format
        self.enable_ttl_cleanup = enable_ttl_cleanup
        self.ttl_cleanup_interval = ttl_cleanup_interval
        
        # Database components
        self.engine = None
        self.Session = None
        self._lock = threading.RLock()
        
        # TTL cleanup
        self._cleanup_thread = None
        self._cleanup_stop_event = threading.Event()
        
        # Connection pool settings
        self.pool_settings = {
            'poolclass': QueuePool,
            'pool_size': pool_size,
            'max_overflow': max_overflow,
            'pool_timeout': pool_timeout,
            'pool_recycle': 3600,  # Recycle connections every hour
            'pool_pre_ping': True   # Validate connections before use
        }
        
        logger.info(f"SQLMemory initialized with database: {self._get_db_type()}")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 12:05:13")
    
    def _get_db_type(self) -> str:
        """Get database type from URL"""
        if self.database_url.startswith('postgresql'):
            return 'PostgreSQL'
        elif self.database_url.startswith('mysql'):
            return 'MySQL'
        elif self.database_url.startswith('sqlite'):
            return 'SQLite'
        elif self.database_url.startswith('mssql'):
            return 'SQL Server'
        elif self.database_url.startswith('oracle'):
            return 'Oracle'
        else:
            return 'Unknown'
    
    def connect(self) -> None:
        """Connect to SQL database and initialize schema"""
        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                self.database_url,
                echo=False,  # Set to True for SQL debugging
                **self.pool_settings
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            # Create tables if they don't exist
            self._create_tables()
            
            # Start TTL cleanup if enabled
            if self.enable_ttl_cleanup:
                self._start_ttl_cleanup()
            
            self._connected = True
            
            logger.info(f"SQL database connected successfully: {self._get_db_type()}")
            logger.info(f"Connected by user: Vatsal216 at 2025-07-22 12:05:13")
            
        except Exception as e:
            logger.error(f"Failed to connect to SQL database: {e}")
            raise MemoryError(
                "SQL database connection failed",
                backend="SQL",
                operation="connect",
                original_error=e
            )
    
    def disconnect(self) -> None:
        """Disconnect from SQL database"""
        try:
            # Stop TTL cleanup
            if self._cleanup_thread:
                self._cleanup_stop_event.set()
                self._cleanup_thread.join(timeout=5)
            
            # Close engine
            if self.engine:
                self.engine.dispose()
                self.engine = None
            
            self.Session = None
            self._connected = False
            
            logger.info("SQL database disconnected successfully")
            
        except Exception as e:
            logger.error(f"Error during SQL disconnect: {e}")
    
    def _create_tables(self) -> None:
        """Create database tables if they don't exist"""
        try:
            # Update table name with prefix
            MemoryItem.__tablename__ = f"{self.table_prefix}memory"
            
            # Create all tables
            Base.metadata.create_all(self.engine)
            
            logger.info("Database tables created/verified successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def _serialize_value(self, value: Any) -> tuple[bytes, str]:
        """Serialize value for storage"""
        if self.serialization_format == 'json':
            try:
                serialized = json.dumps(value).encode('utf-8')
                value_type = 'json'
            except (TypeError, ValueError):
                # Fallback to pickle for non-JSON serializable objects
                serialized = pickle.dumps(value)
                value_type = 'pickle'
        else:
            # Default to pickle
            serialized = pickle.dumps(value)
            value_type = 'pickle'
        
        # Apply compression if enabled and threshold met
        is_compressed = False
        if self.enable_compression and len(serialized) >= self.compression_threshold:
            serialized = gzip.compress(serialized)
            is_compressed = True
        
        return serialized, value_type, is_compressed
    
    def _deserialize_value(self, data: bytes, value_type: str, is_compressed: bool) -> Any:
        """Deserialize value from storage"""
        try:
            # Decompress if needed
            if is_compressed:
                data = gzip.decompress(data)
            
            # Deserialize based on type
            if value_type == 'json':
                return json.loads(data.decode('utf-8'))
            else:
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"Failed to deserialize value: {e}")
            raise MemoryError(
                "Value deserialization failed",
                backend="SQL",
                operation="deserialize",
                original_error=e
            )
    
    def save(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Save value to SQL database with TTL support"""
        def _save():
            session = self.Session()
            try:
                # Serialize value
                serialized_data, value_type, is_compressed = self._serialize_value(value)
                
                # Calculate expiration
                expires_at = None
                if ttl:
                    expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                
                # Check if item exists
                existing_item = session.query(MemoryItem).filter_by(key=key).first()
                
                if existing_item:
                    # Update existing item
                    existing_item.value_data = serialized_data
                    existing_item.value_type = value_type
                    existing_item.expires_at = expires_at
                    existing_item.size_bytes = len(serialized_data)
                    existing_item.is_compressed = is_compressed
                    existing_item.metadata_json = json.dumps({
                        'updated_at': datetime.utcnow().isoformat(),
                        'updated_by': 'Vatsal216'
                    })
                else:
                    # Create new item
                    item = MemoryItem(
                        key=key,
                        value_data=serialized_data,
                        value_type=value_type,
                        expires_at=expires_at,
                        size_bytes=len(serialized_data),
                        is_compressed=is_compressed,
                        metadata_json=json.dumps({
                            'created_by': 'Vatsal216',
                            'created_at': datetime.utcnow().isoformat()
                        })
                    )
                    session.add(item)
                
                session.commit()
                
                logger.debug(f"Saved key '{key}' to SQL database (TTL: {ttl})")
                return True
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"SQL error saving key '{key}': {e}")
                raise MemoryError(f"SQL save failed: {e}")
            finally:
                session.close()
        
        return self._execute_with_metrics(MemoryOperation.SAVE, _save)
    
    def load(self, key: str) -> Any:
        """Load value from SQL database"""
        def _load():
            session = self.Session()
            try:
                item = session.query(MemoryItem).filter_by(key=key).first()
                
                if not item:
                    logger.debug(f"Key '{key}' not found in SQL database")
                    return None
                
                # Check expiration
                if item.expires_at and datetime.utcnow() > item.expires_at:
                    # Remove expired item
                    session.delete(item)
                    session.commit()
                    logger.debug(f"Key '{key}' expired and removed from SQL database")
                    return None
                
                # Update access statistics
                item.access_count += 1
                item.last_access = datetime.utcnow()
                session.commit()
                
                # Deserialize value
                value = self._deserialize_value(
                    item.value_data, 
                    item.value_type, 
                    item.is_compressed
                )
                
                logger.debug(f"Loaded key '{key}' from SQL database")
                return value
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"SQL error loading key '{key}': {e}")
                return None
            finally:
                session.close()
        
        return self._execute_with_metrics(MemoryOperation.LOAD, _load)
    
    def delete(self, key: str) -> bool:
        """Delete value from SQL database"""
        def _delete():
            session = self.Session()
            try:
                item = session.query(MemoryItem).filter_by(key=key).first()
                
                if item:
                    session.delete(item)
                    session.commit()
                    logger.debug(f"Deleted key '{key}' from SQL database")
                    return True
                
                return False
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"SQL error deleting key '{key}': {e}")
                return False
            finally:
                session.close()
        
        return self._execute_with_metrics(MemoryOperation.DELETE, _delete)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in SQL database"""
        def _exists():
            session = self.Session()
            try:
                count = session.query(MemoryItem).filter_by(key=key).count()
                return count > 0
                
            except SQLAlchemyError as e:
                logger.error(f"SQL error checking existence of key '{key}': {e}")
                return False
            finally:
                session.close()
        
        return self._execute_with_metrics(MemoryOperation.EXISTS, _exists)
    
    def clear(self) -> bool:
        """Clear all data from SQL database"""
        def _clear():
            session = self.Session()
            try:
                deleted_count = session.query(MemoryItem).delete()
                session.commit()
                
                logger.info(f"Cleared {deleted_count} items from SQL database")
                return True
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"SQL error clearing database: {e}")
                return False
            finally:
                session.close()
        
        return self._execute_with_metrics(MemoryOperation.CLEAR, _clear)
    
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List all keys in SQL database"""
        def _list_keys():
            session = self.Session()
            try:
                query = session.query(MemoryItem.key)
                
                # Apply pattern filter if provided
                if pattern:
                    # Convert shell pattern to SQL LIKE pattern
                    sql_pattern = pattern.replace('*', '%').replace('?', '_')
                    query = query.filter(MemoryItem.key.like(sql_pattern))
                
                keys = [row.key for row in query.all()]
                return sorted(keys)
                
            except SQLAlchemyError as e:
                logger.error(f"SQL error listing keys: {e}")
                return []
            finally:
                session.close()
        
        return self._execute_with_metrics(MemoryOperation.LIST_KEYS, _list_keys)
    
    def get_size(self) -> int:
        """Get total size of stored data in bytes"""
        def _get_size():
            session = self.Session()
            try:
                total_size = session.query(
                    text("SUM(size_bytes)")
                ).scalar() or 0
                
                return int(total_size)
                
            except SQLAlchemyError as e:
                logger.error(f"SQL error getting size: {e}")
                return 0
            finally:
                session.close()
        
        return self._execute_with_metrics(MemoryOperation.GET_SIZE, _get_size)
    
    def _start_ttl_cleanup(self) -> None:
        """Start background TTL cleanup thread"""
        def cleanup_worker():
            while not self._cleanup_stop_event.wait(self.ttl_cleanup_interval):
                try:
                    self._cleanup_expired_items()
                except Exception as e:
                    logger.error(f"TTL cleanup error: {e}")
        
        self._cleanup_thread = threading.Thread(
            target=cleanup_worker,
            name="SQLMemory-TTL-Cleanup",
            daemon=True
        )
        self._cleanup_thread.start()
        
        logger.info("TTL cleanup thread started")
    
    def _cleanup_expired_items(self) -> int:
        """Clean up expired items from database"""
        session = self.Session()
        try:
            current_time = datetime.utcnow()
            
            # Delete expired items
            deleted_count = session.query(MemoryItem).filter(
                MemoryItem.expires_at.isnot(None),
                MemoryItem.expires_at <= current_time
            ).delete()
            
            session.commit()
            
            if deleted_count > 0:
                logger.debug(f"Cleaned up {deleted_count} expired items")
            
            return deleted_count
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"TTL cleanup error: {e}")
            return 0
        finally:
            session.close()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        session = self.Session()
        try:
            # Basic counts
            total_items = session.query(MemoryItem).count()
            expired_items = session.query(MemoryItem).filter(
                MemoryItem.expires_at.isnot(None),
                MemoryItem.expires_at <= datetime.utcnow()
            ).count()
            
            # Size statistics
            total_size = session.query(text("SUM(size_bytes)")).scalar() or 0
            avg_size = session.query(text("AVG(size_bytes)")).scalar() or 0
            
            # Access statistics
            total_accesses = session.query(text("SUM(access_count)")).scalar() or 0
            avg_accesses = session.query(text("AVG(access_count)")).scalar() or 0
            
            # Compression statistics
            compressed_items = session.query(MemoryItem).filter_by(is_compressed=True).count()
            
            return {
                "total_items": total_items,
                "expired_items": expired_items,
                "active_items": total_items - expired_items,
                "total_size_bytes": int(total_size),
                "average_size_bytes": float(avg_size) if avg_size else 0,
                "total_accesses": int(total_accesses),
                "average_accesses": float(avg_accesses) if avg_accesses else 0,
                "compressed_items": compressed_items,
                "compression_ratio": compressed_items / total_items if total_items > 0 else 0,
                "database_type": self._get_db_type(),
                "table_name": f"{self.table_prefix}memory",
                "created_by": "Vatsal216",
                "timestamp": "2025-07-22 12:05:13"
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}
        finally:
            session.close()
    
    def search_by_metadata(self, metadata_query: Dict[str, Any]) -> List[str]:
        """Search keys by metadata (requires JSON support in database)"""
        session = self.Session()
        try:
            # This is a simplified search - in production you'd want proper JSON querying
            items = session.query(MemoryItem).all()
            matching_keys = []
            
            for item in items:
                try:
                    metadata = json.loads(item.metadata_json or '{}')
                    match = True
                    
                    for key, value in metadata_query.items():
                        if key not in metadata or metadata[key] != value:
                            match = False
                            break
                    
                    if match:
                        matching_keys.append(item.key)
                        
                except (json.JSONDecodeError, KeyError):
                    continue
            
            return sorted(matching_keys)
            
        except SQLAlchemyError as e:
            logger.error(f"Error searching by metadata: {e}")
            return []
        finally:
            session.close()
    
    def backup_to_file(self, filepath: str) -> bool:
        """Backup all data to a file"""
        session = self.Session()
        try:
            items = session.query(MemoryItem).all()
            
            backup_data = {
                'created_at': datetime.utcnow().isoformat(),
                'created_by': 'Vatsal216',
                'database_type': self._get_db_type(),
                'total_items': len(items),
                'items': []
            }
            
            for item in items:
                # Deserialize value for backup
                try:
                    value = self._deserialize_value(
                        item.value_data,
                        item.value_type,
                        item.is_compressed
                    )
                    
                    backup_data['items'].append({
                        'key': item.key,
                        'value': value,
                        'created_at': item.created_at.isoformat(),
                        'expires_at': item.expires_at.isoformat() if item.expires_at else None,
                        'access_count': item.access_count,
                        'metadata': json.loads(item.metadata_json or '{}')
                    })
                except Exception as e:
                    logger.error(f"Failed to backup item '{item.key}': {e}")
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            logger.info(f"Database backed up to {filepath} ({len(items)} items)")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
        finally:
            session.close()
    
    def restore_from_file(self, filepath: str, clear_existing: bool = False) -> bool:
        """Restore data from backup file"""
        try:
            if clear_existing:
                self.clear()
            
            with open(filepath, 'r') as f:
                backup_data = json.load(f)
            
            restored_count = 0
            failed_count = 0
            
            for item_data in backup_data.get('items', []):
                try:
                    key = item_data['key']
                    value = item_data['value']
                    
                    # Calculate TTL if expires_at is set
                    ttl = None
                    if item_data.get('expires_at'):
                        expires_at = datetime.fromisoformat(item_data['expires_at'])
                        if expires_at > datetime.utcnow():
                            ttl = int((expires_at - datetime.utcnow()).total_seconds())
                    
                    # Restore item
                    if self.save(key, value, ttl):
                        restored_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to restore item: {e}")
                    failed_count += 1
            
            logger.info(f"Restore completed: {restored_count} items restored, {failed_count} failed")
            return failed_count == 0
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False