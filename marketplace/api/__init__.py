"""
Integration Marketplace API for CrewGraph AI

Provides RESTful API for marketplace operations including:
- Integration discovery and search
- Installation and management
- Compatibility checking
- Marketplace analytics
- Rating and review system

Author: Vatsal216
Created: 2025-01-27
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from fastapi import FastAPI, HTTPException, Depends, Query, Path as PathParam
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import httpx
    WEB_FRAMEWORK_AVAILABLE = True
except ImportError:
    FastAPI = HTTPException = Depends = Query = PathParam = None
    CORSMiddleware = JSONResponse = BaseModel = Field = None
    httpx = None
    WEB_FRAMEWORK_AVAILABLE = False

from ...types import WorkflowId
from ...utils.logging import get_logger

logger = get_logger(__name__)


class IntegrationCategory(str, Enum):
    """Categories for integrations."""
    DATABASE = "database"
    MESSAGING = "messaging"
    CLOUD_STORAGE = "cloud_storage"
    MONITORING = "monitoring"
    COMMUNICATION = "communication"
    CI_CD = "ci_cd"
    DATA_PROCESSING = "data_processing"
    ML_PLATFORM = "ml_platform"
    SECURITY = "security"
    ANALYTICS = "analytics"


class IntegrationStatus(str, Enum):
    """Status of integrations."""
    AVAILABLE = "available"
    INSTALLED = "installed"
    UPDATING = "updating"
    ERROR = "error"
    DEPRECATED = "deprecated"


class CompatibilityLevel(str, Enum):
    """Compatibility levels."""
    FULL = "full"
    PARTIAL = "partial"
    EXPERIMENTAL = "experimental"
    INCOMPATIBLE = "incompatible"


# Pydantic Models
if WEB_FRAMEWORK_AVAILABLE:
    class IntegrationMetadata(BaseModel):
        """Integration metadata model."""
        id: str = Field(..., description="Unique integration identifier")
        name: str = Field(..., description="Display name")
        description: str = Field(..., description="Integration description")
        version: str = Field(..., description="Current version")
        category: IntegrationCategory = Field(..., description="Integration category")
        author: str = Field(..., description="Author name")
        license: str = Field(default="MIT", description="License type")
        tags: List[str] = Field(default_factory=list, description="Tags for search")
        
        # Technical details
        requirements: List[str] = Field(default_factory=list, description="Python requirements")
        supported_platforms: List[str] = Field(default_factory=list, description="Supported platforms")
        min_crewgraph_version: str = Field(default="1.0.0", description="Minimum CrewGraph version")
        
        # Marketplace info
        downloads: int = Field(default=0, description="Download count")
        rating: float = Field(default=0.0, description="Average rating")
        review_count: int = Field(default=0, description="Number of reviews")
        created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
        updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
        
        # Configuration
        config_schema: Dict[str, Any] = Field(default_factory=dict, description="Configuration schema")
        documentation_url: Optional[str] = Field(None, description="Documentation URL")
        source_url: Optional[str] = Field(None, description="Source code URL")
        
        class Config:
            use_enum_values = True

    class IntegrationInstallRequest(BaseModel):
        """Request to install an integration."""
        integration_id: str = Field(..., description="Integration ID to install")
        version: Optional[str] = Field(None, description="Specific version to install")
        config: Dict[str, Any] = Field(default_factory=dict, description="Configuration parameters")
        
    class IntegrationReview(BaseModel):
        """Integration review model."""
        integration_id: str = Field(..., description="Integration ID")
        user_id: str = Field(..., description="Reviewer user ID")
        rating: int = Field(..., ge=1, le=5, description="Rating 1-5 stars")
        comment: Optional[str] = Field(None, description="Review comment")
        version: str = Field(..., description="Version reviewed")
        created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    class CompatibilityCheck(BaseModel):
        """Compatibility check result."""
        integration_id: str = Field(..., description="Integration ID")
        compatibility_level: CompatibilityLevel = Field(..., description="Compatibility level")
        issues: List[str] = Field(default_factory=list, description="Compatibility issues")
        recommendations: List[str] = Field(default_factory=list, description="Recommendations")
        
    class MarketplaceStats(BaseModel):
        """Marketplace statistics."""
        total_integrations: int = Field(..., description="Total number of integrations")
        total_downloads: int = Field(..., description="Total downloads")
        active_installations: int = Field(..., description="Active installations")
        popular_categories: List[Dict[str, Any]] = Field(..., description="Popular categories")
        trending_integrations: List[str] = Field(..., description="Trending integration IDs")

else:
    # Fallback dataclasses when FastAPI is not available
    @dataclass
    class IntegrationMetadata:
        id: str
        name: str
        description: str
        version: str
        category: str
        author: str
        license: str = "MIT"
        tags: List[str] = field(default_factory=list)
        requirements: List[str] = field(default_factory=list)
        supported_platforms: List[str] = field(default_factory=list)
        min_crewgraph_version: str = "1.0.0"
        downloads: int = 0
        rating: float = 0.0
        review_count: int = 0
        created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
        updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
        config_schema: Dict[str, Any] = field(default_factory=dict)
        documentation_url: Optional[str] = None
        source_url: Optional[str] = None


class IntegrationRegistry:
    """
    Registry for managing integration metadata and discovery.
    """
    
    def __init__(self, registry_dir: Optional[str] = None):
        """Initialize integration registry."""
        self.registry_dir = Path(registry_dir) if registry_dir else Path("marketplace/registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.integrations: Dict[str, IntegrationMetadata] = {}
        self.registry_file = self.registry_dir / "integrations.json"
        
        # Load existing registry
        self._load_registry()
        
        # Initialize with default integrations if empty
        if not self.integrations:
            self._initialize_default_integrations()
    
    def _load_registry(self):
        """Load integration registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                for integration_data in data.get('integrations', []):
                    integration = IntegrationMetadata(**integration_data)
                    self.integrations[integration.id] = integration
                    
                logger.info(f"Loaded {len(self.integrations)} integrations from registry")
                
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
    
    def _save_registry(self):
        """Save integration registry to disk."""
        try:
            data = {
                "version": "1.0.0",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "integrations": [
                    integration.__dict__ if hasattr(integration, '__dict__') 
                    else integration.dict() if hasattr(integration, 'dict')
                    else vars(integration)
                    for integration in self.integrations.values()
                ]
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            logger.info(f"Saved {len(self.integrations)} integrations to registry")
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def register_integration(self, integration: IntegrationMetadata) -> bool:
        """Register a new integration."""
        try:
            # Validate integration
            if not integration.id or not integration.name:
                raise ValueError("Integration ID and name are required")
            
            # Check for duplicates
            if integration.id in self.integrations:
                raise ValueError(f"Integration {integration.id} already exists")
            
            # Add to registry
            self.integrations[integration.id] = integration
            self._save_registry()
            
            logger.info(f"Registered integration: {integration.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register integration {integration.id}: {e}")
            return False
    
    def get_integration(self, integration_id: str) -> Optional[IntegrationMetadata]:
        """Get integration by ID."""
        return self.integrations.get(integration_id)
    
    def search_integrations(
        self,
        query: Optional[str] = None,
        category: Optional[IntegrationCategory] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[IntegrationMetadata]:
        """Search for integrations."""
        results = list(self.integrations.values())
        
        # Filter by category
        if category:
            results = [i for i in results if i.category == category.value]
        
        # Filter by tags
        if tags:
            results = [i for i in results if any(tag in i.tags for tag in tags)]
        
        # Filter by query (search in name, description, tags)
        if query:
            query_lower = query.lower()
            results = [
                i for i in results 
                if (query_lower in i.name.lower() or 
                    query_lower in i.description.lower() or
                    any(query_lower in tag.lower() for tag in i.tags))
            ]
        
        # Sort by popularity (downloads and rating)
        results.sort(key=lambda x: (x.downloads * x.rating), reverse=True)
        
        # Apply pagination
        return results[offset:offset + limit]
    
    def get_popular_integrations(self, limit: int = 10) -> List[IntegrationMetadata]:
        """Get most popular integrations."""
        integrations = list(self.integrations.values())
        integrations.sort(key=lambda x: x.downloads, reverse=True)
        return integrations[:limit]
    
    def get_trending_integrations(self, limit: int = 10) -> List[IntegrationMetadata]:
        """Get trending integrations (simplified implementation)."""
        # In a real implementation, this would consider download velocity
        integrations = list(self.integrations.values())
        integrations.sort(key=lambda x: (x.downloads, x.rating), reverse=True)
        return integrations[:limit]
    
    def update_integration_stats(
        self,
        integration_id: str,
        downloads: Optional[int] = None,
        rating: Optional[float] = None,
        review_count: Optional[int] = None
    ):
        """Update integration statistics."""
        if integration_id in self.integrations:
            integration = self.integrations[integration_id]
            
            if downloads is not None:
                integration.downloads = downloads
            if rating is not None:
                integration.rating = rating
            if review_count is not None:
                integration.review_count = review_count
            
            integration.updated_at = datetime.now(timezone.utc).isoformat()
            self._save_registry()
            
            logger.info(f"Updated stats for integration: {integration_id}")
    
    def _initialize_default_integrations(self):
        """Initialize registry with default integrations."""
        default_integrations = [
            # Database integrations
            IntegrationMetadata(
                id="postgresql",
                name="PostgreSQL",
                description="PostgreSQL database integration with advanced features",
                version="1.0.0",
                category="database",
                author="CrewGraph AI Team",
                tags=["database", "sql", "postgresql"],
                requirements=["psycopg2-binary>=2.9.0"],
                supported_platforms=["linux", "darwin", "win32"],
                downloads=1250,
                rating=4.8,
                review_count=45,
                config_schema={
                    "host": {"type": "string", "required": True},
                    "port": {"type": "integer", "default": 5432},
                    "database": {"type": "string", "required": True},
                    "username": {"type": "string", "required": True},
                    "password": {"type": "string", "required": True}
                }
            ),
            IntegrationMetadata(
                id="mongodb",
                name="MongoDB",
                description="MongoDB NoSQL database integration",
                version="1.2.0",
                category="database",
                author="CrewGraph AI Team",
                tags=["database", "nosql", "mongodb"],
                requirements=["pymongo>=4.0.0"],
                supported_platforms=["linux", "darwin", "win32"],
                downloads=980,
                rating=4.6,
                review_count=32,
                config_schema={
                    "uri": {"type": "string", "required": True},
                    "database": {"type": "string", "required": True}
                }
            ),
            IntegrationMetadata(
                id="redis",
                name="Redis",
                description="Redis in-memory data structure store integration",
                version="1.1.0",
                category="database",
                author="CrewGraph AI Team",
                tags=["database", "cache", "redis"],
                requirements=["redis>=4.0.0"],
                supported_platforms=["linux", "darwin", "win32"],
                downloads=1450,
                rating=4.9,
                review_count=67,
                config_schema={
                    "host": {"type": "string", "required": True},
                    "port": {"type": "integer", "default": 6379},
                    "password": {"type": "string", "required": False}
                }
            ),
            
            # Messaging integrations
            IntegrationMetadata(
                id="rabbitmq",
                name="RabbitMQ",
                description="RabbitMQ message broker integration",
                version="1.0.0",
                category="messaging",
                author="CrewGraph AI Team",
                tags=["messaging", "queue", "rabbitmq"],
                requirements=["pika>=1.3.0"],
                supported_platforms=["linux", "darwin", "win32"],
                downloads=720,
                rating=4.5,
                review_count=28,
                config_schema={
                    "host": {"type": "string", "required": True},
                    "port": {"type": "integer", "default": 5672},
                    "username": {"type": "string", "required": True},
                    "password": {"type": "string", "required": True}
                }
            ),
            IntegrationMetadata(
                id="kafka",
                name="Apache Kafka",
                description="Apache Kafka distributed streaming platform integration",
                version="1.1.0",
                category="messaging",
                author="CrewGraph AI Team",
                tags=["messaging", "streaming", "kafka"],
                requirements=["kafka-python>=2.0.0"],
                supported_platforms=["linux", "darwin", "win32"],
                downloads=890,
                rating=4.7,
                review_count=41,
                config_schema={
                    "bootstrap_servers": {"type": "array", "required": True},
                    "topic": {"type": "string", "required": True}
                }
            ),
            
            # Cloud Storage integrations
            IntegrationMetadata(
                id="aws_s3",
                name="AWS S3",
                description="Amazon S3 cloud storage integration",
                version="1.3.0",
                category="cloud_storage",
                author="CrewGraph AI Team",
                tags=["cloud", "storage", "aws", "s3"],
                requirements=["boto3>=1.26.0"],
                supported_platforms=["linux", "darwin", "win32"],
                downloads=1680,
                rating=4.8,
                review_count=89,
                config_schema={
                    "access_key": {"type": "string", "required": True},
                    "secret_key": {"type": "string", "required": True},
                    "region": {"type": "string", "required": True},
                    "bucket": {"type": "string", "required": True}
                }
            ),
            IntegrationMetadata(
                id="azure_blob",
                name="Azure Blob Storage",
                description="Microsoft Azure Blob Storage integration",
                version="1.0.0",
                category="cloud_storage",
                author="CrewGraph AI Team",
                tags=["cloud", "storage", "azure", "blob"],
                requirements=["azure-storage-blob>=12.14.0"],
                supported_platforms=["linux", "darwin", "win32"],
                downloads=650,
                rating=4.4,
                review_count=23,
                config_schema={
                    "connection_string": {"type": "string", "required": True},
                    "container": {"type": "string", "required": True}
                }
            ),
            IntegrationMetadata(
                id="gcp_storage",
                name="Google Cloud Storage",
                description="Google Cloud Storage integration",
                version="1.1.0",
                category="cloud_storage",
                author="CrewGraph AI Team",
                tags=["cloud", "storage", "gcp", "google"],
                requirements=["google-cloud-storage>=2.7.0"],
                supported_platforms=["linux", "darwin", "win32"],
                downloads=580,
                rating=4.6,
                review_count=19,
                config_schema={
                    "credentials_path": {"type": "string", "required": True},
                    "bucket": {"type": "string", "required": True}
                }
            ),
            
            # Monitoring integrations
            IntegrationMetadata(
                id="prometheus",
                name="Prometheus",
                description="Prometheus monitoring and alerting integration",
                version="1.2.0",
                category="monitoring",
                author="CrewGraph AI Team",
                tags=["monitoring", "metrics", "prometheus"],
                requirements=["prometheus-client>=0.16.0"],
                supported_platforms=["linux", "darwin", "win32"],
                downloads=1120,
                rating=4.7,
                review_count=54,
                config_schema={
                    "endpoint": {"type": "string", "required": True},
                    "pushgateway_url": {"type": "string", "required": False}
                }
            ),
            IntegrationMetadata(
                id="datadog",
                name="Datadog",
                description="Datadog monitoring and analytics integration",
                version="1.0.0",
                category="monitoring",
                author="CrewGraph AI Team",
                tags=["monitoring", "analytics", "datadog"],
                requirements=["datadog>=0.44.0"],
                supported_platforms=["linux", "darwin", "win32"],
                downloads=740,
                rating=4.5,
                review_count=31,
                config_schema={
                    "api_key": {"type": "string", "required": True},
                    "app_key": {"type": "string", "required": True}
                }
            ),
            
            # Communication integrations
            IntegrationMetadata(
                id="slack",
                name="Slack",
                description="Slack messaging and notifications integration",
                version="1.1.0",
                category="communication",
                author="CrewGraph AI Team",
                tags=["communication", "slack", "notifications"],
                requirements=["slack-sdk>=3.19.0"],
                supported_platforms=["linux", "darwin", "win32"],
                downloads=1340,
                rating=4.9,
                review_count=78,
                config_schema={
                    "token": {"type": "string", "required": True},
                    "channel": {"type": "string", "required": True}
                }
            ),
            IntegrationMetadata(
                id="discord",
                name="Discord",
                description="Discord bot and notifications integration",
                version="1.0.0",
                category="communication",
                author="CrewGraph AI Team",
                tags=["communication", "discord", "notifications"],
                requirements=["discord.py>=2.1.0"],
                supported_platforms=["linux", "darwin", "win32"],
                downloads=520,
                rating=4.3,
                review_count=22,
                config_schema={
                    "token": {"type": "string", "required": True},
                    "channel_id": {"type": "string", "required": True}
                }
            )
        ]
        
        for integration in default_integrations:
            self.integrations[integration.id] = integration
        
        self._save_registry()
        logger.info(f"Initialized registry with {len(default_integrations)} default integrations")


class CompatibilityChecker:
    """
    Checks compatibility between integrations and the current environment.
    """
    
    def __init__(self):
        """Initialize compatibility checker."""
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        import platform
        import sys
        
        return {
            "platform": platform.system().lower(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "architecture": platform.machine(),
            "crewgraph_version": "1.0.0"  # This should come from actual version
        }
    
    def check_compatibility(
        self,
        integration: IntegrationMetadata
    ) -> CompatibilityCheck:
        """Check compatibility for an integration."""
        issues = []
        recommendations = []
        compatibility_level = CompatibilityLevel.FULL
        
        # Check platform compatibility
        if integration.supported_platforms:
            if self.system_info["platform"] not in integration.supported_platforms:
                issues.append(f"Platform {self.system_info['platform']} is not officially supported")
                compatibility_level = CompatibilityLevel.EXPERIMENTAL
        
        # Check CrewGraph version compatibility
        if integration.min_crewgraph_version:
            # Simple version comparison (in production, use proper version parsing)
            min_version = integration.min_crewgraph_version
            current_version = self.system_info["crewgraph_version"]
            
            if min_version > current_version:
                issues.append(f"Requires CrewGraph version {min_version} or higher (current: {current_version})")
                compatibility_level = CompatibilityLevel.INCOMPATIBLE
        
        # Check Python requirements (simplified)
        missing_requirements = self._check_requirements(integration.requirements)
        if missing_requirements:
            issues.extend([f"Missing requirement: {req}" for req in missing_requirements])
            if compatibility_level == CompatibilityLevel.FULL:
                compatibility_level = CompatibilityLevel.PARTIAL
            recommendations.append("Install missing requirements with: pip install " + " ".join(missing_requirements))
        
        return CompatibilityCheck(
            integration_id=integration.id,
            compatibility_level=compatibility_level,
            issues=issues,
            recommendations=recommendations
        )
    
    def _check_requirements(self, requirements: List[str]) -> List[str]:
        """Check which requirements are missing (simplified implementation)."""
        missing = []
        
        for requirement in requirements:
            # Extract package name (ignore version specifiers for simplicity)
            package_name = requirement.split(">=")[0].split("==")[0].split(">")[0].split("<")[0]
            
            try:
                __import__(package_name.replace("-", "_"))
            except ImportError:
                missing.append(requirement)
        
        return missing


class MarketplaceAPI:
    """
    RESTful API for the integration marketplace.
    """
    
    def __init__(self, registry: IntegrationRegistry):
        """Initialize marketplace API."""
        self.registry = registry
        self.compatibility_checker = CompatibilityChecker()
        self.reviews: Dict[str, List[Dict]] = {}  # In production, use a database
        
        if WEB_FRAMEWORK_AVAILABLE:
            self.app = FastAPI(
                title="CrewGraph AI Integration Marketplace",
                description="API for discovering, installing, and managing integrations",
                version="1.0.0"
            )
            
            # Add CORS middleware
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        if not WEB_FRAMEWORK_AVAILABLE:
            return
        
        @self.app.get("/integrations", response_model=List[IntegrationMetadata])
        async def list_integrations(
            query: Optional[str] = Query(None, description="Search query"),
            category: Optional[IntegrationCategory] = Query(None, description="Filter by category"),
            tags: Optional[str] = Query(None, description="Comma-separated tags"),
            limit: int = Query(50, description="Number of results"),
            offset: int = Query(0, description="Offset for pagination")
        ):
            """List and search integrations."""
            tag_list = tags.split(",") if tags else None
            return self.registry.search_integrations(query, category, tag_list, limit, offset)
        
        @self.app.get("/integrations/{integration_id}", response_model=IntegrationMetadata)
        async def get_integration(integration_id: str = PathParam(..., description="Integration ID")):
            """Get specific integration details."""
            integration = self.registry.get_integration(integration_id)
            if not integration:
                raise HTTPException(status_code=404, detail="Integration not found")
            return integration
        
        @self.app.get("/integrations/{integration_id}/compatibility", response_model=CompatibilityCheck)
        async def check_integration_compatibility(integration_id: str = PathParam(..., description="Integration ID")):
            """Check integration compatibility."""
            integration = self.registry.get_integration(integration_id)
            if not integration:
                raise HTTPException(status_code=404, detail="Integration not found")
            
            return self.compatibility_checker.check_compatibility(integration)
        
        @self.app.post("/integrations/{integration_id}/install")
        async def install_integration(
            integration_id: str = PathParam(..., description="Integration ID"),
            request: IntegrationInstallRequest
        ):
            """Install an integration."""
            integration = self.registry.get_integration(integration_id)
            if not integration:
                raise HTTPException(status_code=404, detail="Integration not found")
            
            # Check compatibility first
            compatibility = self.compatibility_checker.check_compatibility(integration)
            if compatibility.compatibility_level == CompatibilityLevel.INCOMPATIBLE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Integration is incompatible: {', '.join(compatibility.issues)}"
                )
            
            # Simulate installation process
            installation_id = str(uuid.uuid4())
            
            # Update download count
            self.registry.update_integration_stats(integration_id, downloads=integration.downloads + 1)
            
            return {
                "installation_id": installation_id,
                "status": "installed",
                "integration_id": integration_id,
                "version": request.version or integration.version,
                "installed_at": datetime.now(timezone.utc).isoformat()
            }
        
        @self.app.get("/integrations/popular", response_model=List[IntegrationMetadata])
        async def get_popular_integrations(limit: int = Query(10, description="Number of results")):
            """Get popular integrations."""
            return self.registry.get_popular_integrations(limit)
        
        @self.app.get("/integrations/trending", response_model=List[IntegrationMetadata])
        async def get_trending_integrations(limit: int = Query(10, description="Number of results")):
            """Get trending integrations."""
            return self.registry.get_trending_integrations(limit)
        
        @self.app.post("/integrations/{integration_id}/reviews")
        async def create_review(
            integration_id: str = PathParam(..., description="Integration ID"),
            review: IntegrationReview
        ):
            """Create a review for an integration."""
            integration = self.registry.get_integration(integration_id)
            if not integration:
                raise HTTPException(status_code=404, detail="Integration not found")
            
            # Store review
            if integration_id not in self.reviews:
                self.reviews[integration_id] = []
            
            review_data = review.dict() if hasattr(review, 'dict') else vars(review)
            review_data["id"] = str(uuid.uuid4())
            self.reviews[integration_id].append(review_data)
            
            # Update integration rating
            reviews = self.reviews[integration_id]
            avg_rating = sum(r["rating"] for r in reviews) / len(reviews)
            self.registry.update_integration_stats(
                integration_id,
                rating=avg_rating,
                review_count=len(reviews)
            )
            
            return {"message": "Review created successfully", "review_id": review_data["id"]}
        
        @self.app.get("/integrations/{integration_id}/reviews")
        async def get_reviews(
            integration_id: str = PathParam(..., description="Integration ID"),
            limit: int = Query(50, description="Number of results"),
            offset: int = Query(0, description="Offset for pagination")
        ):
            """Get reviews for an integration."""
            integration = self.registry.get_integration(integration_id)
            if not integration:
                raise HTTPException(status_code=404, detail="Integration not found")
            
            reviews = self.reviews.get(integration_id, [])
            return reviews[offset:offset + limit]
        
        @self.app.get("/marketplace/stats", response_model=MarketplaceStats)
        async def get_marketplace_stats():
            """Get marketplace statistics."""
            integrations = list(self.registry.integrations.values())
            
            total_downloads = sum(i.downloads for i in integrations)
            category_stats = {}
            
            for integration in integrations:
                category = integration.category
                if category not in category_stats:
                    category_stats[category] = {"count": 0, "downloads": 0}
                category_stats[category]["count"] += 1
                category_stats[category]["downloads"] += integration.downloads
            
            popular_categories = [
                {"category": cat, **stats}
                for cat, stats in sorted(
                    category_stats.items(),
                    key=lambda x: x[1]["downloads"],
                    reverse=True
                )
            ]
            
            trending = self.registry.get_trending_integrations(5)
            
            return MarketplaceStats(
                total_integrations=len(integrations),
                total_downloads=total_downloads,
                active_installations=total_downloads,  # Simplified
                popular_categories=popular_categories,
                trending_integrations=[i.id for i in trending]
            )
    
    def get_app(self):
        """Get FastAPI app instance."""
        return getattr(self, 'app', None)


# Export all classes
__all__ = [
    "IntegrationCategory",
    "IntegrationStatus",
    "CompatibilityLevel",
    "IntegrationMetadata",
    "IntegrationRegistry",
    "CompatibilityChecker",
    "MarketplaceAPI"
]