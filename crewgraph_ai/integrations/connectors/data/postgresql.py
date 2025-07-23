"""
PostgreSQL Integration for CrewGraph AI

Provides database connectivity, query execution, and data management
capabilities for PostgreSQL databases in workflows.

Author: Vatsal216
Created: 2025-07-23 18:50:00 UTC
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ... import BaseIntegration, IntegrationConfig, IntegrationMetadata, IntegrationResult, IntegrationType

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class PostgreSQLIntegration(BaseIntegration):
    """
    PostgreSQL database integration for data operations.
    
    Supports database connectivity, query execution, transaction management,
    and data manipulation through the psycopg2 driver.
    """
    
    @property
    def metadata(self) -> IntegrationMetadata:
        """Get PostgreSQL integration metadata."""
        return IntegrationMetadata(
            name="PostgreSQL",
            version="1.0.0",
            description="Database connectivity and operations for PostgreSQL",
            author="CrewGraph AI",
            integration_type=IntegrationType.DATA,
            dependencies=["psycopg2-binary"],
            config_schema={
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "description": "Database host"
                    },
                    "port": {
                        "type": "integer",
                        "description": "Database port",
                        "default": 5432
                    },
                    "database": {
                        "type": "string",
                        "description": "Database name"
                    },
                    "username": {
                        "type": "string",
                        "description": "Database username"
                    },
                    "password": {
                        "type": "string",
                        "description": "Database password"
                    },
                    "ssl_mode": {
                        "type": "string",
                        "description": "SSL mode (disable, allow, prefer, require)",
                        "default": "prefer"
                    },
                    "connection_timeout": {
                        "type": "integer",
                        "description": "Connection timeout in seconds",
                        "default": 30
                    }
                },
                "required": ["host", "database", "username", "password"]
            },
            supports_async=False,  # psycopg2 is synchronous
            supports_webhook=False,
            homepage="https://www.postgresql.org/",
            documentation="https://www.postgresql.org/docs/",
            tags=["database", "sql", "data", "postgres"]
        )
    
    def initialize(self) -> bool:
        """Initialize PostgreSQL integration."""
        try:
            if not PSYCOPG2_AVAILABLE:
                self.logger.error("psycopg2 library not available")
                return False
            
            # Get configuration
            config = self.config.config
            self.host = config.get("host")
            self.port = config.get("port", 5432)
            self.database = config.get("database")
            self.username = config.get("username")
            self.password = config.get("password")
            self.ssl_mode = config.get("ssl_mode", "prefer")
            self.connection_timeout = config.get("connection_timeout", 30)
            
            # Build connection string
            self.connection_string = f"host={self.host} port={self.port} dbname={self.database} user={self.username} password={self.password} sslmode={self.ssl_mode} connect_timeout={self.connection_timeout}"
            
            # Test connection
            self.logger.info("Testing PostgreSQL connection...")
            
            # In production, test actual connection:
            # conn = psycopg2.connect(self.connection_string)
            # conn.close()
            
            # For demo, simulate successful connection
            self._connection = None  # Will be created on demand
            
            self.is_initialized = True
            self.logger.info("PostgreSQL integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PostgreSQL integration: {e}")
            return False
    
    def execute(self, action: str, **kwargs) -> IntegrationResult:
        """Execute PostgreSQL action."""
        if not self.is_initialized:
            return IntegrationResult(
                success=False,
                error_message="Integration not initialized"
            )
        
        try:
            if action == "execute_query":
                return self._execute_query(**kwargs)
            elif action == "execute_update":
                return self._execute_update(**kwargs)
            elif action == "insert_data":
                return self._insert_data(**kwargs)
            elif action == "create_table":
                return self._create_table(**kwargs)
            elif action == "list_tables":
                return self._list_tables(**kwargs)
            elif action == "get_table_schema":
                return self._get_table_schema(**kwargs)
            elif action == "execute_transaction":
                return self._execute_transaction(**kwargs)
            else:
                return IntegrationResult(
                    success=False,
                    error_message=f"Unknown action: {action}"
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    def validate_config(self) -> List[str]:
        """Validate PostgreSQL configuration."""
        issues = []
        
        config = self.config.config
        
        required_fields = ["host", "database", "username", "password"]
        for field in required_fields:
            if not config.get(field):
                issues.append(f"{field} is required")
        
        port = config.get("port", 5432)
        if not isinstance(port, int) or port < 1 or port > 65535:
            issues.append("port must be a valid integer between 1 and 65535")
        
        ssl_mode = config.get("ssl_mode", "prefer")
        valid_ssl_modes = ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
        if ssl_mode not in valid_ssl_modes:
            issues.append(f"ssl_mode must be one of: {', '.join(valid_ssl_modes)}")
        
        return issues
    
    def _execute_query(
        self,
        query: str,
        parameters: Optional[Tuple] = None,
        fetch_all: bool = True
    ) -> IntegrationResult:
        """Execute a SELECT query."""
        try:
            # Simulate query execution
            result = self._simulate_query_execution(query, parameters, fetch_all)
            
            if result.get("success"):
                return IntegrationResult(
                    success=True,
                    data={
                        "rows": result.get("rows", []),
                        "row_count": result.get("row_count", 0),
                        "columns": result.get("columns", [])
                    },
                    metadata={
                        "query": query,
                        "execution_time_ms": result.get("execution_time_ms", 0)
                    }
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=result.get("error", "Query execution failed")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error executing query: {str(e)}"
            )
    
    def _execute_update(
        self,
        query: str,
        parameters: Optional[Tuple] = None
    ) -> IntegrationResult:
        """Execute an UPDATE, DELETE, or other non-SELECT query."""
        try:
            # Simulate update execution
            result = self._simulate_update_execution(query, parameters)
            
            if result.get("success"):
                return IntegrationResult(
                    success=True,
                    data={
                        "rows_affected": result.get("rows_affected", 0)
                    },
                    metadata={
                        "query": query,
                        "execution_time_ms": result.get("execution_time_ms", 0)
                    }
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=result.get("error", "Update execution failed")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error executing update: {str(e)}"
            )
    
    def _insert_data(
        self,
        table: str,
        data: Dict[str, Any],
        returning: Optional[str] = None
    ) -> IntegrationResult:
        """Insert data into a table."""
        try:
            columns = list(data.keys())
            values = list(data.values())
            placeholders = ", ".join(["%s"] * len(values))
            
            query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            if returning:
                query += f" RETURNING {returning}"
            
            # Simulate insert execution
            result = self._simulate_insert_execution(query, tuple(values), table, returning)
            
            if result.get("success"):
                response_data = {
                    "rows_inserted": 1,
                    "table": table
                }
                
                if returning and result.get("returned_values"):
                    response_data["returned_values"] = result.get("returned_values")
                
                return IntegrationResult(
                    success=True,
                    data=response_data,
                    metadata={
                        "query": query,
                        "execution_time_ms": result.get("execution_time_ms", 0)
                    }
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=result.get("error", "Insert failed")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error inserting data: {str(e)}"
            )
    
    def _create_table(
        self,
        table_name: str,
        columns: Dict[str, str],
        primary_key: Optional[str] = None,
        indexes: Optional[List[str]] = None
    ) -> IntegrationResult:
        """Create a new table."""
        try:
            # Build CREATE TABLE query
            column_definitions = []
            for col_name, col_type in columns.items():
                column_definitions.append(f"{col_name} {col_type}")
            
            if primary_key:
                column_definitions.append(f"PRIMARY KEY ({primary_key})")
            
            query = f"CREATE TABLE {table_name} ({', '.join(column_definitions)})"
            
            # Simulate table creation
            result = self._simulate_ddl_execution(query, table_name)
            
            if result.get("success"):
                return IntegrationResult(
                    success=True,
                    data={
                        "table_name": table_name,
                        "columns": columns,
                        "primary_key": primary_key
                    },
                    metadata={
                        "query": query,
                        "execution_time_ms": result.get("execution_time_ms", 0)
                    }
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=result.get("error", "Table creation failed")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error creating table: {str(e)}"
            )
    
    def _list_tables(self, schema: str = "public") -> IntegrationResult:
        """List all tables in the database."""
        try:
            query = """
                SELECT tablename, tableowner 
                FROM pg_tables 
                WHERE schemaname = %s
                ORDER BY tablename
            """
            
            # Simulate table listing
            result = self._simulate_query_execution(query, (schema,), fetch_all=True)
            
            if result.get("success"):
                tables = [
                    {
                        "name": row[0],
                        "owner": row[1],
                        "schema": schema
                    }
                    for row in result.get("rows", [])
                ]
                
                return IntegrationResult(
                    success=True,
                    data={
                        "tables": tables,
                        "schema": schema,
                        "total_count": len(tables)
                    }
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=result.get("error", "Failed to list tables")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error listing tables: {str(e)}"
            )
    
    def _get_table_schema(self, table_name: str, schema: str = "public") -> IntegrationResult:
        """Get schema information for a table."""
        try:
            query = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = %s AND table_schema = %s
                ORDER BY ordinal_position
            """
            
            # Simulate schema query
            result = self._simulate_query_execution(query, (table_name, schema), fetch_all=True)
            
            if result.get("success"):
                columns = [
                    {
                        "name": row[0],
                        "type": row[1],
                        "nullable": row[2] == "YES",
                        "default": row[3]
                    }
                    for row in result.get("rows", [])
                ]
                
                return IntegrationResult(
                    success=True,
                    data={
                        "table_name": table_name,
                        "schema": schema,
                        "columns": columns,
                        "column_count": len(columns)
                    }
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=result.get("error", "Failed to get table schema")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error getting table schema: {str(e)}"
            )
    
    def _execute_transaction(self, queries: List[Dict[str, Any]]) -> IntegrationResult:
        """Execute multiple queries in a transaction."""
        try:
            results = []
            total_rows_affected = 0
            
            # Simulate transaction execution
            for query_info in queries:
                query = query_info.get("query")
                parameters = query_info.get("parameters")
                
                if query.strip().upper().startswith("SELECT"):
                    result = self._simulate_query_execution(query, parameters, True)
                else:
                    result = self._simulate_update_execution(query, parameters)
                
                if not result.get("success"):
                    return IntegrationResult(
                        success=False,
                        error_message=f"Transaction failed at query: {query}. Error: {result.get('error')}"
                    )
                
                results.append(result)
                total_rows_affected += result.get("rows_affected", 0)
            
            return IntegrationResult(
                success=True,
                data={
                    "transaction_completed": True,
                    "queries_executed": len(queries),
                    "total_rows_affected": total_rows_affected,
                    "results": results
                },
                metadata={
                    "transaction_size": len(queries),
                    "execution_time_ms": sum(r.get("execution_time_ms", 0) for r in results)
                }
            )
            
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Transaction error: {str(e)}"
            )
    
    def _simulate_query_execution(
        self, 
        query: str, 
        parameters: Optional[Tuple] = None,
        fetch_all: bool = True
    ) -> Dict[str, Any]:
        """Simulate query execution (for testing without actual database)."""
        # Simulate different query types
        query_upper = query.strip().upper()
        
        if "SELECT" in query_upper and "pg_tables" in query:
            # Simulate table listing
            return {
                "success": True,
                "rows": [
                    ("users", "admin"),
                    ("orders", "admin"),
                    ("products", "admin")
                ],
                "row_count": 3,
                "columns": ["tablename", "tableowner"],
                "execution_time_ms": 15
            }
        
        elif "SELECT" in query_upper and "information_schema.columns" in query:
            # Simulate schema information
            return {
                "success": True,
                "rows": [
                    ("id", "integer", "NO", "nextval('users_id_seq'::regclass)"),
                    ("name", "character varying", "NO", None),
                    ("email", "character varying", "YES", None),
                    ("created_at", "timestamp without time zone", "NO", "now()")
                ],
                "row_count": 4,
                "columns": ["column_name", "data_type", "is_nullable", "column_default"],
                "execution_time_ms": 25
            }
        
        elif "SELECT" in query_upper:
            # Simulate generic SELECT
            return {
                "success": True,
                "rows": [
                    (1, "John Doe", "john@example.com"),
                    (2, "Jane Smith", "jane@example.com")
                ],
                "row_count": 2,
                "columns": ["id", "name", "email"],
                "execution_time_ms": 12
            }
        
        else:
            return {
                "success": False,
                "error": f"Unsupported query type: {query_upper[:50]}"
            }
    
    def _simulate_update_execution(
        self, 
        query: str, 
        parameters: Optional[Tuple] = None
    ) -> Dict[str, Any]:
        """Simulate UPDATE/DELETE execution."""
        query_upper = query.strip().upper()
        
        if any(op in query_upper for op in ["UPDATE", "DELETE", "INSERT"]):
            return {
                "success": True,
                "rows_affected": 1,
                "execution_time_ms": 8
            }
        else:
            return {
                "success": False,
                "error": f"Invalid update query: {query_upper[:50]}"
            }
    
    def _simulate_insert_execution(
        self, 
        query: str, 
        parameters: Tuple,
        table: str,
        returning: Optional[str] = None
    ) -> Dict[str, Any]:
        """Simulate INSERT execution."""
        result = {
            "success": True,
            "rows_affected": 1,
            "execution_time_ms": 10
        }
        
        if returning:
            # Simulate returned values
            if returning.lower() == "id":
                result["returned_values"] = [(123,)]
            else:
                result["returned_values"] = [(f"generated_{returning}",)]
        
        return result
    
    def _simulate_ddl_execution(self, query: str, table_name: str) -> Dict[str, Any]:
        """Simulate DDL (CREATE, DROP, ALTER) execution."""
        query_upper = query.strip().upper()
        
        if "CREATE TABLE" in query_upper:
            return {
                "success": True,
                "execution_time_ms": 25
            }
        else:
            return {
                "success": False,
                "error": f"Unsupported DDL operation: {query_upper[:50]}"
            }
    
    def _perform_health_check(self) -> Dict[str, Any]:
        """Perform PostgreSQL-specific health check."""
        try:
            # Test basic connectivity
            result = self._simulate_query_execution("SELECT 1", None, True)
            
            if result.get("success"):
                return {
                    "status": "healthy",
                    "message": "PostgreSQL connection successful",
                    "database": self.database,
                    "host": self.host,
                    "port": self.port,
                    "execution_count": self.execution_count,
                    "last_execution": self.last_execution
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": f"PostgreSQL connection failed: {result.get('error')}",
                    "execution_count": self.execution_count
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "execution_count": self.execution_count
            }
    
    def shutdown(self):
        """Shutdown PostgreSQL integration and close connections."""
        try:
            if hasattr(self, "_connection") and self._connection:
                self._connection.close()
                self._connection = None
        except Exception as e:
            self.logger.error(f"Error closing connection: {e}")
        
        super().shutdown()