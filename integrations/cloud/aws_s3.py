"""
AWS S3 Integration for CrewGraph AI

Provides comprehensive Amazon S3 cloud storage integration with:
- File upload and download operations
- Bucket management
- Object metadata handling
- Presigned URL generation
- Lifecycle management

Author: CrewGraph AI Team
Version: 1.3.0
"""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    boto3 = ClientError = NoCredentialsError = None
    AWS_AVAILABLE = False

from ....marketplace.plugins import BasePlugin, PluginContext
from ....utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class S3Config:
    """S3 integration configuration."""
    
    access_key: str
    secret_key: str
    region: str
    bucket: str
    
    # Optional settings
    endpoint_url: Optional[str] = None
    signature_version: str = "s3v4"
    max_retries: int = 3
    timeout: int = 300


class AWSS3Integration(BasePlugin):
    """
    AWS S3 cloud storage integration plugin.
    
    Provides comprehensive S3 operations for file storage,
    data archival, and workflow artifact management.
    """
    
    def __init__(self, context: PluginContext):
        """Initialize S3 integration."""
        super().__init__(context)
        
        self.config = self._parse_config()
        self.s3_client = None
        self.s3_resource = None
        self.operation_stats = {
            "uploads": 0,
            "downloads": 0,
            "deletes": 0,
            "errors": 0,
            "bytes_uploaded": 0,
            "bytes_downloaded": 0
        }
    
    def _parse_config(self) -> S3Config:
        """Parse plugin configuration."""
        config_data = self.context.config
        
        return S3Config(
            access_key=config_data.get("access_key"),
            secret_key=config_data.get("secret_key"),
            region=config_data.get("region"),
            bucket=config_data.get("bucket"),
            endpoint_url=config_data.get("endpoint_url"),
            signature_version=config_data.get("signature_version", "s3v4"),
            max_retries=config_data.get("max_retries", 3),
            timeout=config_data.get("timeout", 300)
        )
    
    async def initialize(self) -> bool:
        """Initialize S3 client."""
        if not AWS_AVAILABLE:
            self.context.log_error("boto3 library not available")
            return False
        
        if not all([self.config.access_key, self.config.secret_key, 
                   self.config.region, self.config.bucket]):
            self.context.log_error("Missing required S3 configuration")
            return False
        
        try:
            # Create S3 client
            session = boto3.Session(
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                region_name=self.config.region
            )
            
            self.s3_client = session.client(
                's3',
                endpoint_url=self.config.endpoint_url,
                config=boto3.session.Config(
                    signature_version=self.config.signature_version,
                    retries={'max_attempts': self.config.max_retries},
                    read_timeout=self.config.timeout
                )
            )
            
            self.s3_resource = session.resource('s3')
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.config.bucket)
            
            self.context.log_info(f"Connected to S3 bucket: {self.config.bucket}")
            return True
            
        except NoCredentialsError:
            self.context.log_error("AWS credentials not found")
            return False
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                self.context.log_error(f"S3 bucket not found: {self.config.bucket}")
            else:
                self.context.log_error(f"S3 connection failed: {e}")
            return False
        except Exception as e:
            self.context.log_error(f"Failed to initialize S3: {e}")
            return False
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute S3 operations.
        
        Supported operations:
        - upload_file: Upload files to S3
        - download_file: Download files from S3
        - delete_object: Delete objects from S3
        - list_objects: List objects in bucket
        - create_presigned_url: Generate presigned URLs
        - copy_object: Copy objects within/between buckets
        - get_object_metadata: Get object metadata
        """
        operation = task.get("operation")
        
        if operation == "upload_file":
            return await self._upload_file(task)
        elif operation == "download_file":
            return await self._download_file(task)
        elif operation == "delete_object":
            return await self._delete_object(task)
        elif operation == "list_objects":
            return await self._list_objects(task)
        elif operation == "create_presigned_url":
            return await self._create_presigned_url(task)
        elif operation == "copy_object":
            return await self._copy_object(task)
        elif operation == "get_object_metadata":
            return await self._get_object_metadata(task)
        elif operation == "upload_data":
            return await self._upload_data(task)
        else:
            return {
                "success": False,
                "error": f"Unsupported operation: {operation}"
            }
    
    async def _upload_file(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Upload a file to S3."""
        local_path = task.get("local_path")
        s3_key = task.get("s3_key")
        metadata = task.get("metadata", {})
        content_type = task.get("content_type")
        bucket = task.get("bucket", self.config.bucket)
        
        if not local_path or not s3_key:
            return {"success": False, "error": "local_path and s3_key are required"}
        
        local_file = Path(local_path)
        if not local_file.exists():
            return {"success": False, "error": f"Local file not found: {local_path}"}
        
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            if content_type:
                extra_args['ContentType'] = content_type
            
            start_time = time.time()
            file_size = local_file.stat().st_size
            
            # Upload file
            self.s3_client.upload_file(
                str(local_file),
                bucket,
                s3_key,
                ExtraArgs=extra_args
            )
            
            upload_time = time.time() - start_time
            
            # Update stats
            self.operation_stats["uploads"] += 1
            self.operation_stats["bytes_uploaded"] += file_size
            
            return {
                "success": True,
                "s3_key": s3_key,
                "bucket": bucket,
                "file_size": file_size,
                "upload_time": upload_time,
                "s3_url": f"s3://{bucket}/{s3_key}"
            }
            
        except ClientError as e:
            self.operation_stats["errors"] += 1
            error_msg = e.response['Error']['Message']
            self.context.log_error(f"S3 upload failed: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "local_path": local_path,
                "s3_key": s3_key
            }
    
    async def _upload_data(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Upload data content directly to S3."""
        data = task.get("data")
        s3_key = task.get("s3_key")
        metadata = task.get("metadata", {})
        content_type = task.get("content_type", "application/octet-stream")
        bucket = task.get("bucket", self.config.bucket)
        
        if data is None or not s3_key:
            return {"success": False, "error": "data and s3_key are required"}
        
        try:
            # Convert data to bytes if needed
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
                if not content_type or content_type == "application/octet-stream":
                    content_type = "text/plain"
            elif isinstance(data, dict):
                data_bytes = json.dumps(data).encode('utf-8')
                if not content_type or content_type == "application/octet-stream":
                    content_type = "application/json"
            else:
                data_bytes = data
            
            start_time = time.time()
            data_size = len(data_bytes)
            
            # Upload data
            self.s3_client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=data_bytes,
                ContentType=content_type,
                Metadata=metadata
            )
            
            upload_time = time.time() - start_time
            
            # Update stats
            self.operation_stats["uploads"] += 1
            self.operation_stats["bytes_uploaded"] += data_size
            
            return {
                "success": True,
                "s3_key": s3_key,
                "bucket": bucket,
                "data_size": data_size,
                "upload_time": upload_time,
                "content_type": content_type,
                "s3_url": f"s3://{bucket}/{s3_key}"
            }
            
        except ClientError as e:
            self.operation_stats["errors"] += 1
            error_msg = e.response['Error']['Message']
            self.context.log_error(f"S3 data upload failed: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "s3_key": s3_key
            }
    
    async def _download_file(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Download a file from S3."""
        s3_key = task.get("s3_key")
        local_path = task.get("local_path")
        bucket = task.get("bucket", self.config.bucket)
        
        if not s3_key or not local_path:
            return {"success": False, "error": "s3_key and local_path are required"}
        
        try:
            start_time = time.time()
            
            # Download file
            self.s3_client.download_file(bucket, s3_key, local_path)
            
            download_time = time.time() - start_time
            
            # Get file size
            local_file = Path(local_path)
            file_size = local_file.stat().st_size if local_file.exists() else 0
            
            # Update stats
            self.operation_stats["downloads"] += 1
            self.operation_stats["bytes_downloaded"] += file_size
            
            return {
                "success": True,
                "s3_key": s3_key,
                "local_path": local_path,
                "file_size": file_size,
                "download_time": download_time
            }
            
        except ClientError as e:
            self.operation_stats["errors"] += 1
            error_msg = e.response['Error']['Message']
            self.context.log_error(f"S3 download failed: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "s3_key": s3_key,
                "local_path": local_path
            }
    
    async def _delete_object(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Delete an object from S3."""
        s3_key = task.get("s3_key")
        bucket = task.get("bucket", self.config.bucket)
        
        if not s3_key:
            return {"success": False, "error": "s3_key is required"}
        
        try:
            # Delete object
            self.s3_client.delete_object(Bucket=bucket, Key=s3_key)
            
            # Update stats
            self.operation_stats["deletes"] += 1
            
            return {
                "success": True,
                "s3_key": s3_key,
                "bucket": bucket
            }
            
        except ClientError as e:
            self.operation_stats["errors"] += 1
            error_msg = e.response['Error']['Message']
            self.context.log_error(f"S3 delete failed: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "s3_key": s3_key
            }
    
    async def _list_objects(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """List objects in S3 bucket."""
        prefix = task.get("prefix", "")
        max_keys = task.get("max_keys", 1000)
        bucket = task.get("bucket", self.config.bucket)
        
        try:
            # List objects
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            objects = []
            total_size = 0
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    objects.append({
                        "key": obj['Key'],
                        "size": obj['Size'],
                        "last_modified": obj['LastModified'].isoformat(),
                        "etag": obj['ETag'].strip('"'),
                        "storage_class": obj.get('StorageClass', 'STANDARD')
                    })
                    total_size += obj['Size']
            
            return {
                "success": True,
                "objects": objects,
                "object_count": len(objects),
                "total_size": total_size,
                "prefix": prefix,
                "is_truncated": response.get('IsTruncated', False)
            }
            
        except ClientError as e:
            self.operation_stats["errors"] += 1
            error_msg = e.response['Error']['Message']
            self.context.log_error(f"S3 list failed: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "prefix": prefix
            }
    
    async def _create_presigned_url(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create a presigned URL for S3 object."""
        s3_key = task.get("s3_key")
        expiration = task.get("expiration", 3600)  # 1 hour default
        http_method = task.get("http_method", "GET")
        bucket = task.get("bucket", self.config.bucket)
        
        if not s3_key:
            return {"success": False, "error": "s3_key is required"}
        
        try:
            # Generate presigned URL
            if http_method.upper() == "GET":
                url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket, 'Key': s3_key},
                    ExpiresIn=expiration
                )
            elif http_method.upper() == "PUT":
                url = self.s3_client.generate_presigned_url(
                    'put_object',
                    Params={'Bucket': bucket, 'Key': s3_key},
                    ExpiresIn=expiration
                )
            else:
                return {"success": False, "error": f"Unsupported HTTP method: {http_method}"}
            
            return {
                "success": True,
                "presigned_url": url,
                "s3_key": s3_key,
                "expiration": expiration,
                "http_method": http_method.upper()
            }
            
        except ClientError as e:
            self.operation_stats["errors"] += 1
            error_msg = e.response['Error']['Message']
            self.context.log_error(f"Presigned URL generation failed: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "s3_key": s3_key
            }
    
    async def _copy_object(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Copy an object within S3."""
        source_key = task.get("source_key")
        dest_key = task.get("dest_key")
        source_bucket = task.get("source_bucket", self.config.bucket)
        dest_bucket = task.get("dest_bucket", self.config.bucket)
        
        if not source_key or not dest_key:
            return {"success": False, "error": "source_key and dest_key are required"}
        
        try:
            # Copy object
            copy_source = {'Bucket': source_bucket, 'Key': source_key}
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=dest_bucket,
                Key=dest_key
            )
            
            return {
                "success": True,
                "source_key": source_key,
                "dest_key": dest_key,
                "source_bucket": source_bucket,
                "dest_bucket": dest_bucket
            }
            
        except ClientError as e:
            self.operation_stats["errors"] += 1
            error_msg = e.response['Error']['Message']
            self.context.log_error(f"S3 copy failed: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "source_key": source_key,
                "dest_key": dest_key
            }
    
    async def _get_object_metadata(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for an S3 object."""
        s3_key = task.get("s3_key")
        bucket = task.get("bucket", self.config.bucket)
        
        if not s3_key:
            return {"success": False, "error": "s3_key is required"}
        
        try:
            # Get object metadata
            response = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
            
            metadata = {
                "content_length": response.get('ContentLength', 0),
                "content_type": response.get('ContentType', ''),
                "last_modified": response.get('LastModified', '').isoformat() if response.get('LastModified') else '',
                "etag": response.get('ETag', '').strip('"'),
                "storage_class": response.get('StorageClass', 'STANDARD'),
                "metadata": response.get('Metadata', {}),
                "cache_control": response.get('CacheControl', ''),
                "expires": response.get('Expires', '').isoformat() if response.get('Expires') else ''
            }
            
            return {
                "success": True,
                "s3_key": s3_key,
                "metadata": metadata
            }
            
        except ClientError as e:
            self.operation_stats["errors"] += 1
            error_msg = e.response['Error']['Message']
            self.context.log_error(f"Get metadata failed: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "s3_key": s3_key
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            if not self.s3_client:
                return {
                    "status": "unhealthy",
                    "error": "S3 client not initialized"
                }
            
            # Test bucket access
            self.s3_client.head_bucket(Bucket=self.config.bucket)
            
            return {
                "status": "healthy",
                "bucket": self.config.bucket,
                "region": self.config.region,
                "statistics": self.operation_stats
            }
            
        except ClientError as e:
            return {
                "status": "unhealthy",
                "error": e.response['Error']['Message']
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def cleanup(self):
        """Cleanup S3 resources."""
        self.s3_client = None
        self.s3_resource = None
        self.context.log_info("S3 integration cleaned up")


# Plugin entry point
Plugin = AWSS3Integration

# Plugin manifest
PLUGIN_MANIFEST = {
    "id": "aws_s3",
    "name": "AWS S3 Integration",
    "version": "1.3.0", 
    "description": "Comprehensive Amazon S3 cloud storage integration with file operations and metadata management",
    "author": "CrewGraph AI Team",
    "api_version": "1.0.0",
    "min_crewgraph_version": "1.0.0",
    "python_version": ">=3.8",
    "dependencies": ["boto3>=1.26.0"],
    "entry_point": "s3_integration.py",
    "plugin_class": "AWSS3Integration",
    "category": "cloud_storage",
    "tags": ["cloud", "storage", "aws", "s3", "files"],
    "permissions": ["network_access", "file_access"],
    "sandbox_enabled": True,
    "network_access": True,
    "file_access": ["data", "temp"],
    "config_schema": {
        "access_key": {"type": "string", "required": True, "description": "AWS Access Key ID"},
        "secret_key": {"type": "string", "required": True, "description": "AWS Secret Access Key"},
        "region": {"type": "string", "required": True, "description": "AWS Region"},
        "bucket": {"type": "string", "required": True, "description": "S3 Bucket Name"},
        "endpoint_url": {"type": "string", "required": False, "description": "Custom S3 endpoint URL"},
        "signature_version": {"type": "string", "default": "s3v4", "description": "S3 signature version"},
        "max_retries": {"type": "integer", "default": 3, "description": "Maximum retry attempts"},
        "timeout": {"type": "integer", "default": 300, "description": "Request timeout in seconds"}
    }
}