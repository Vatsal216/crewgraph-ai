"""
PostgreSQL Integration for CrewGraph AI

Provides comprehensive PostgreSQL database integration with:
- Connection management and pooling
- Query execution and transaction handling
- Data pipeline integration
- Schema migration support
- Performance monitoring

Author: CrewGraph AI Team
Version: 1.0.0
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

try:
    import psycopg2
    from psycopg2 import pool, sql
    from psycopg2.extras import RealDictCursor
    POSTGRESQL_AVAILABLE = True
except ImportError:
    psycopg2 = pool = sql = RealDictCursor = None
    POSTGRESQL_AVAILABLE = False

from ....marketplace.plugins import BasePlugin, PluginContext
from ....utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PostgreSQLConfig:
    """PostgreSQL connection configuration."""
    
    host: str
    port: int = 5432
    database: str
    username: str
    password: str
    
    # Connection pool settings
    min_connections: int = 1
    max_connections: int = 20
    
    # Advanced settings
    ssl_mode: str = "prefer"
    timeout: int = 30
    retry_attempts: int = 3
    
    def to_connection_string(self) -> str:
        """Convert to PostgreSQL connection string."""
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}&connect_timeout={self.timeout}"
        )


class PostgreSQLIntegration(BasePlugin):
    """
    PostgreSQL database integration plugin.
    
    Provides database operations, connection pooling, and data management
    capabilities for CrewGraph AI workflows.
    """
    
    def __init__(self, context: PluginContext):
        """Initialize PostgreSQL integration."""
        super().__init__(context)
        
        self.config = self._parse_config()
        self.connection_pool = None
        self.connection_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_connections": 0,
            "active_connections": 0
        }
    
    def _parse_config(self) -> PostgreSQLConfig:
        """Parse plugin configuration."""
        config_data = self.context.config
        
        return PostgreSQLConfig(
            host=config_data.get("host", "localhost"),
            port=config_data.get("port", 5432),
            database=config_data.get("database"),
            username=config_data.get("username"),
            password=config_data.get("password"),
            min_connections=config_data.get("min_connections", 1),
            max_connections=config_data.get("max_connections", 20),
            ssl_mode=config_data.get("ssl_mode", "prefer"),
            timeout=config_data.get("timeout", 30),
            retry_attempts=config_data.get("retry_attempts", 3)
        )
    
    async def initialize(self) -> bool:
        """Initialize PostgreSQL connection pool."""
        if not POSTGRESQL_AVAILABLE:
            self.context.log_error("psycopg2 library not available")
            return False
        
        if not all([self.config.host, self.config.database, 
                   self.config.username, self.config.password]):
            self.context.log_error("Missing required PostgreSQL configuration")
            return False
        
        try:
            # Create connection pool
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                sslmode=self.config.ssl_mode,
                connect_timeout=self.config.timeout
            )
            
            # Test connection
            conn = self.connection_pool.getconn()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                self.context.log_info(f"Connected to PostgreSQL: {version}")
                cursor.close()
            finally:
                self.connection_pool.putconn(conn)
            
            self.context.log_info("PostgreSQL integration initialized successfully")
            return True
            
        except Exception as e:
            self.context.log_error(f"Failed to initialize PostgreSQL: {e}")
            return False
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute PostgreSQL database operations.
        
        Supported operations:
        - query: Execute SELECT queries
        - execute: Execute INSERT/UPDATE/DELETE queries
        - transaction: Execute multiple queries in a transaction
        - create_table: Create database tables
        - migrate: Run database migrations
        """
        operation = task.get("operation")
        
        if operation == "query":
            return await self._execute_query(task)
        elif operation == "execute":
            return await self._execute_command(task)
        elif operation == "transaction":
            return await self._execute_transaction(task)
        elif operation == "create_table":
            return await self._create_table(task)
        elif operation == "migrate":
            return await self._run_migration(task)
        elif operation == "bulk_insert":
            return await self._bulk_insert(task)
        else:
            return {
                "success": False,
                "error": f"Unsupported operation: {operation}"
            }
    
    async def _execute_query(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a SELECT query."""
        query = task.get("query")
        params = task.get("params", {})
        fetch_mode = task.get("fetch_mode", "all")  # all, one, many
        limit = task.get("limit")
        
        if not query:
            return {"success": False, "error": "Query is required"}
        
        try:
            conn = self.connection_pool.getconn()
            
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                # Add limit if specified
                if limit and "LIMIT" not in query.upper():
                    query += f" LIMIT {limit}"
                
                start_time = time.time()
                cursor.execute(query, params)
                
                # Fetch results based on mode
                if fetch_mode == "one":
                    result = cursor.fetchone()
                    data = dict(result) if result else None
                elif fetch_mode == "many":
                    many_count = task.get("many_count", 100)
                    results = cursor.fetchmany(many_count)
                    data = [dict(row) for row in results]
                else:  # fetch_mode == "all"
                    results = cursor.fetchall()
                    data = [dict(row) for row in results]
                
                execution_time = time.time() - start_time
                
                cursor.close()
                self.connection_stats["total_queries"] += 1
                self.connection_stats["successful_queries"] += 1
                
                return {
                    "success": True,
                    "data": data,
                    "row_count": len(data) if isinstance(data, list) else (1 if data else 0),
                    "execution_time": execution_time,
                    "query": query
                }
                
            finally:
                self.connection_pool.putconn(conn)
                
        except Exception as e:
            self.connection_stats["total_queries"] += 1
            self.connection_stats["failed_queries"] += 1
            self.context.log_error(f"Query execution failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def _execute_command(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an INSERT/UPDATE/DELETE command."""
        query = task.get("query")
        params = task.get("params", {})
        return_generated = task.get("return_generated", False)
        
        if not query:
            return {"success": False, "error": "Query is required"}
        
        try:
            conn = self.connection_pool.getconn()
            
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                start_time = time.time()
                cursor.execute(query, params)
                
                rows_affected = cursor.rowcount
                generated_data = None
                
                # Get generated data if requested (e.g., RETURNING clause)
                if return_generated and cursor.description:
                    generated_data = cursor.fetchall()
                    generated_data = [dict(row) for row in generated_data]
                
                conn.commit()
                execution_time = time.time() - start_time
                
                cursor.close()
                self.connection_stats["total_queries"] += 1
                self.connection_stats["successful_queries"] += 1
                
                result = {
                    "success": True,
                    "rows_affected": rows_affected,
                    "execution_time": execution_time,
                    "query": query
                }
                
                if generated_data:
                    result["generated_data"] = generated_data
                
                return result
                
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                self.connection_pool.putconn(conn)
                
        except Exception as e:
            self.connection_stats["total_queries"] += 1
            self.connection_stats["failed_queries"] += 1
            self.context.log_error(f"Command execution failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def _execute_transaction(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multiple queries in a transaction."""
        queries = task.get("queries", [])
        
        if not queries:
            return {"success": False, "error": "Queries list is required"}
        
        try:
            conn = self.connection_pool.getconn()
            
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                results = []
                
                start_time = time.time()
                
                for i, query_info in enumerate(queries):
                    query = query_info.get("query")
                    params = query_info.get("params", {})
                    
                    if not query:
                        raise ValueError(f"Query {i} is missing")
                    
                    cursor.execute(query, params)
                    
                    result = {
                        "query_index": i,
                        "rows_affected": cursor.rowcount,
                        "query": query
                    }
                    
                    # Fetch results if it's a SELECT
                    if cursor.description and query.strip().upper().startswith("SELECT"):
                        data = cursor.fetchall()
                        result["data"] = [dict(row) for row in data]
                    
                    results.append(result)
                
                conn.commit()
                execution_time = time.time() - start_time
                
                cursor.close()
                self.connection_stats["total_queries"] += len(queries)
                self.connection_stats["successful_queries"] += len(queries)
                
                return {
                    "success": True,
                    "results": results,
                    "total_queries": len(queries),
                    "execution_time": execution_time
                }
                
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                self.connection_pool.putconn(conn)
                
        except Exception as e:
            self.connection_stats["total_queries"] += len(queries)
            self.connection_stats["failed_queries"] += len(queries)
            self.context.log_error(f"Transaction execution failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "queries": [q.get("query") for q in queries]
            }
    
    async def _create_table(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create a database table."""
        table_name = task.get("table_name")
        columns = task.get("columns", [])
        if_not_exists = task.get("if_not_exists", True)
        
        if not table_name or not columns:
            return {
                "success": False,
                "error": "table_name and columns are required"
            }
        
        try:
            # Build CREATE TABLE query
            column_definitions = []
            for col in columns:
                col_def = f'{col["name"]} {col["type"]}'
                
                if col.get("primary_key"):
                    col_def += " PRIMARY KEY"
                if col.get("not_null"):
                    col_def += " NOT NULL"
                if col.get("unique"):
                    col_def += " UNIQUE"
                if col.get("default"):
                    col_def += f' DEFAULT {col["default"]}'
                
                column_definitions.append(col_def)
            
            query = f"CREATE TABLE "
            if if_not_exists:
                query += "IF NOT EXISTS "
            
            query += f"{table_name} ({', '.join(column_definitions)})"
            
            # Add table constraints if specified
            constraints = task.get("constraints", [])
            if constraints:
                query += f", {', '.join(constraints)}"
            
            return await self._execute_command({
                "query": query,
                "params": {}
            })
            
        except Exception as e:
            self.context.log_error(f"Table creation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "table_name": table_name
            }
    
    async def _bulk_insert(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform bulk insert operations."""
        table_name = task.get("table_name")
        data = task.get("data", [])
        columns = task.get("columns")
        on_conflict = task.get("on_conflict", "")
        
        if not table_name or not data:
            return {
                "success": False,
                "error": "table_name and data are required"
            }
        
        try:
            conn = self.connection_pool.getconn()
            
            try:
                cursor = conn.cursor()
                
                # Determine columns if not provided
                if not columns and data:
                    columns = list(data[0].keys())
                
                # Build INSERT query
                placeholders = ", ".join(["%s"] * len(columns))
                query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                
                if on_conflict:
                    query += f" {on_conflict}"
                
                # Prepare data for bulk insert
                values = []
                for row in data:
                    if isinstance(row, dict):
                        values.append([row.get(col) for col in columns])
                    else:
                        values.append(row)
                
                start_time = time.time()
                cursor.executemany(query, values)
                
                rows_affected = cursor.rowcount
                conn.commit()
                execution_time = time.time() - start_time
                
                cursor.close()
                self.connection_stats["total_queries"] += 1
                self.connection_stats["successful_queries"] += 1
                
                return {
                    "success": True,
                    "rows_affected": rows_affected,
                    "execution_time": execution_time,
                    "records_processed": len(data)
                }
                
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                self.connection_pool.putconn(conn)
                
        except Exception as e:
            self.connection_stats["total_queries"] += 1
            self.connection_stats["failed_queries"] += 1
            self.context.log_error(f"Bulk insert failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "table_name": table_name,
                "records_attempted": len(data)
            }
    
    async def _run_migration(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run database migrations."""
        migration_sql = task.get("migration_sql")
        migration_name = task.get("migration_name", "unnamed_migration")
        
        if not migration_sql:
            return {
                "success": False,
                "error": "migration_sql is required"
            }
        
        try:
            # Execute migration in a transaction
            return await self._execute_transaction({
                "queries": [
                    {"query": migration_sql, "params": {}}
                ]
            })
            
        except Exception as e:
            self.context.log_error(f"Migration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "migration_name": migration_name
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            if not self.connection_pool:
                return {
                    "status": "unhealthy",
                    "error": "Connection pool not initialized"
                }
            
            # Test connection
            conn = self.connection_pool.getconn()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                
                # Get connection pool stats
                pool_stats = {
                    "min_connections": self.config.min_connections,
                    "max_connections": self.config.max_connections,
                    "active_connections": self.connection_stats["active_connections"]
                }
                
                return {
                    "status": "healthy",
                    "database": self.config.database,
                    "host": self.config.host,
                    "connection_pool": pool_stats,
                    "statistics": self.connection_stats
                }
                
            finally:
                self.connection_pool.putconn(conn)
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def cleanup(self):
        """Cleanup PostgreSQL resources."""
        if self.connection_pool:
            try:
                self.connection_pool.closeall()
                self.context.log_info("PostgreSQL connection pool closed")
            except Exception as e:
                self.context.log_error(f"Error closing connection pool: {e}")


# Plugin entry point (required for plugin system)
Plugin = PostgreSQLIntegration


# Plugin manifest for marketplace
PLUGIN_MANIFEST = {
    "id": "postgresql",
    "name": "PostgreSQL Integration",
    "version": "1.0.0",
    "description": "Comprehensive PostgreSQL database integration with connection pooling and advanced features",
    "author": "CrewGraph AI Team",
    "api_version": "1.0.0",
    "min_crewgraph_version": "1.0.0",
    "python_version": ">=3.8",
    "dependencies": ["psycopg2-binary>=2.9.0"],
    "entry_point": "postgresql_integration.py",
    "plugin_class": "PostgreSQLIntegration",
    "category": "database",
    "tags": ["database", "sql", "postgresql", "data"],
    "permissions": ["network_access", "database_access"],
    "sandbox_enabled": True,
    "network_access": True,
    "file_access": ["data"],
    "config_schema": {
        "host": {"type": "string", "required": True, "description": "PostgreSQL host"},
        "port": {"type": "integer", "default": 5432, "description": "PostgreSQL port"},
        "database": {"type": "string", "required": True, "description": "Database name"},
        "username": {"type": "string", "required": True, "description": "Username"},
        "password": {"type": "string", "required": True, "description": "Password"},
        "min_connections": {"type": "integer", "default": 1, "description": "Minimum pool connections"},
        "max_connections": {"type": "integer", "default": 20, "description": "Maximum pool connections"},
        "ssl_mode": {"type": "string", "default": "prefer", "description": "SSL mode"},
        "timeout": {"type": "integer", "default": 30, "description": "Connection timeout"},
        "retry_attempts": {"type": "integer", "default": 3, "description": "Retry attempts"}
    }
}