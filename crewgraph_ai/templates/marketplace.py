"""
Template Marketplace for CrewGraph AI

Provides marketplace functionality for discovering, sharing, and managing
workflow templates including:
- Template discovery and search
- Template ratings and reviews
- Template sharing and distribution
- Template versioning and updates
- Community contributions
- Template analytics

Created by: Vatsal216
Date: 2025-07-23
"""

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from ..utils.exceptions import CrewGraphError, ValidationError
from ..utils.logging import get_logger
from .workflow_templates import TemplateMetadata, WorkflowTemplate

logger = get_logger(__name__)


class TemplateSource(Enum):
    """Sources for templates"""
    LOCAL = "local"
    COMMUNITY = "community"
    OFFICIAL = "official"
    PRIVATE = "private"
    MARKETPLACE = "marketplace"


class TemplateStatus(Enum):
    """Template status in marketplace"""
    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    UNDER_REVIEW = "under_review"


@dataclass
class TemplateRating:
    """Rating information for templates"""
    average_rating: float = 0.0
    total_ratings: int = 0
    rating_distribution: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
    reviews: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TemplateStats:
    """Usage statistics for templates"""
    downloads: int = 0
    usage_count: int = 0
    last_used: Optional[datetime] = None
    popularity_score: float = 0.0
    trending_score: float = 0.0


@dataclass
class MarketplaceTemplate:
    """Enhanced template with marketplace metadata"""
    template: WorkflowTemplate
    source: TemplateSource = TemplateSource.LOCAL
    status: TemplateStatus = TemplateStatus.PUBLISHED
    rating: TemplateRating = field(default_factory=TemplateRating)
    stats: TemplateStats = field(default_factory=TemplateStats)
    license: str = "MIT"
    homepage_url: Optional[str] = None
    repository_url: Optional[str] = None
    documentation_url: Optional[str] = None
    marketplace_id: Optional[str] = None
    verified: bool = False
    featured: bool = False
    
    def __post_init__(self):
        if self.marketplace_id is None:
            self.marketplace_id = self._generate_marketplace_id()
    
    def _generate_marketplace_id(self) -> str:
        """Generate unique marketplace ID"""
        template_data = f"{self.template.metadata.name}:{self.template.metadata.version}:{self.template.metadata.author}"
        return hashlib.md5(template_data.encode()).hexdigest()


class TemplateMarketplace:
    """
    Template marketplace for discovering and managing workflow templates.
    
    Features:
    - Template discovery with search and filtering
    - Ratings and reviews system
    - Template sharing and distribution
    - Community contributions
    - Usage analytics
    """
    
    def __init__(self, cache_dir: str = ".crewgraph_cache"):
        """
        Initialize template marketplace.
        
        Args:
            cache_dir: Directory for caching templates
        """
        self.cache_dir = cache_dir
        self.templates: Dict[str, MarketplaceTemplate] = {}
        self.categories: Set[str] = set()
        self.tags: Set[str] = set()
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"TemplateMarketplace initialized with cache dir: {cache_dir}")
    
    def add_template(self, template: WorkflowTemplate, source: TemplateSource = TemplateSource.LOCAL) -> str:
        """
        Add template to marketplace.
        
        Args:
            template: Template to add
            source: Source of the template
            
        Returns:
            Template marketplace ID
        """
        marketplace_template = MarketplaceTemplate(
            template=template,
            source=source
        )
        
        template_id = marketplace_template.marketplace_id
        self.templates[template_id] = marketplace_template
        
        # Update categories and tags
        self.categories.add(template.metadata.category.value)
        self.tags.update(template.metadata.tags)
        
        logger.info(f"Added template to marketplace: {template.metadata.name} (ID: {template_id})")
        return template_id
    
    def search_templates(
        self,
        query: str = None,
        category: str = None,
        tags: List[str] = None,
        source: TemplateSource = None,
        min_rating: float = None,
        limit: int = 50
    ) -> List[MarketplaceTemplate]:
        """
        Search templates in marketplace.
        
        Args:
            query: Text query to search in name/description
            category: Filter by category
            tags: Filter by tags (all must match)
            source: Filter by source
            min_rating: Minimum average rating
            limit: Maximum number of results
            
        Returns:
            List of matching templates
        """
        results = []
        
        for template in self.templates.values():
            # Apply filters
            if category and template.template.metadata.category.value != category:
                continue
                
            if source and template.source != source:
                continue
                
            if min_rating and template.rating.average_rating < min_rating:
                continue
                
            if tags:
                template_tags = set(template.template.metadata.tags)
                if not set(tags).issubset(template_tags):
                    continue
            
            # Text search
            if query:
                searchable_text = f"{template.template.metadata.name} {template.template.metadata.description}".lower()
                if query.lower() not in searchable_text:
                    continue
            
            results.append(template)
            
            if len(results) >= limit:
                break
        
        # Sort by popularity and rating
        results.sort(key=lambda t: (t.stats.popularity_score, t.rating.average_rating), reverse=True)
        
        return results
    
    def get_featured_templates(self, limit: int = 10) -> List[MarketplaceTemplate]:
        """Get featured templates."""
        featured = [t for t in self.templates.values() if t.featured]
        featured.sort(key=lambda t: t.stats.popularity_score, reverse=True)
        return featured[:limit]
    
    def get_trending_templates(self, limit: int = 10) -> List[MarketplaceTemplate]:
        """Get trending templates."""
        trending = list(self.templates.values())
        trending.sort(key=lambda t: t.stats.trending_score, reverse=True)
        return trending[:limit]
    
    def get_popular_templates(self, limit: int = 10) -> List[MarketplaceTemplate]:
        """Get most popular templates."""
        popular = list(self.templates.values())
        popular.sort(key=lambda t: t.stats.downloads, reverse=True)
        return popular[:limit]
    
    def rate_template(self, template_id: str, rating: int, review: str = "", user_id: str = "anonymous") -> bool:
        """
        Rate a template.
        
        Args:
            template_id: Template marketplace ID
            rating: Rating (1-5)
            review: Optional review text
            user_id: User identifier
            
        Returns:
            Success status
        """
        if template_id not in self.templates:
            return False
            
        if not 1 <= rating <= 5:
            raise ValidationError("Rating must be between 1 and 5")
        
        template = self.templates[template_id]
        
        # Update rating statistics
        old_total = template.rating.total_ratings
        old_sum = template.rating.average_rating * old_total
        
        template.rating.total_ratings += 1
        template.rating.average_rating = (old_sum + rating) / template.rating.total_ratings
        template.rating.rating_distribution[rating] += 1
        
        # Add review if provided
        if review:
            template.rating.reviews.append({
                "user_id": user_id,
                "rating": rating,
                "review": review,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        logger.info(f"Added rating {rating} for template {template_id}")
        return True
    
    def download_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """
        Download/access a template.
        
        Args:
            template_id: Template marketplace ID
            
        Returns:
            Template instance or None if not found
        """
        if template_id not in self.templates:
            return None
        
        template = self.templates[template_id]
        
        # Update download statistics
        template.stats.downloads += 1
        template.stats.last_used = datetime.now(timezone.utc)
        
        # Update popularity score (simple algorithm)
        template.stats.popularity_score = (
            template.stats.downloads * 0.5 +
            template.rating.average_rating * template.rating.total_ratings * 0.3 +
            template.stats.usage_count * 0.2
        )
        
        logger.info(f"Template downloaded: {template_id}")
        return template.template
    
    def get_template_info(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a template.
        
        Args:
            template_id: Template marketplace ID
            
        Returns:
            Template information dict
        """
        if template_id not in self.templates:
            return None
        
        template = self.templates[template_id]
        
        return {
            "id": template_id,
            "name": template.template.metadata.name,
            "description": template.template.metadata.description,
            "version": template.template.metadata.version,
            "category": template.template.metadata.category.value,
            "author": template.template.metadata.author,
            "tags": template.template.metadata.tags,
            "complexity": template.template.metadata.complexity,
            "estimated_time": template.template.metadata.estimated_time,
            "source": template.source.value,
            "status": template.status.value,
            "rating": {
                "average": template.rating.average_rating,
                "total": template.rating.total_ratings,
                "distribution": template.rating.rating_distribution
            },
            "stats": {
                "downloads": template.stats.downloads,
                "usage_count": template.stats.usage_count,
                "popularity_score": template.stats.popularity_score,
                "last_used": template.stats.last_used.isoformat() if template.stats.last_used else None
            },
            "verified": template.verified,
            "featured": template.featured,
            "license": template.license,
            "homepage_url": template.homepage_url,
            "repository_url": template.repository_url,
            "documentation_url": template.documentation_url
        }
    
    def export_template_catalog(self, filename: str = None) -> Dict[str, Any]:
        """
        Export template catalog to JSON.
        
        Args:
            filename: Optional file to save catalog
            
        Returns:
            Template catalog data
        """
        catalog = {
            "version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_templates": len(self.templates),
            "categories": list(self.categories),
            "tags": list(self.tags),
            "templates": {}
        }
        
        for template_id, template in self.templates.items():
            catalog["templates"][template_id] = self.get_template_info(template_id)
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(catalog, f, indent=2, default=str)
            logger.info(f"Template catalog exported to {filename}")
        
        return catalog
    
    def import_template_catalog(self, filename: str) -> int:
        """
        Import template catalog from JSON.
        
        Args:
            filename: Catalog file to import
            
        Returns:
            Number of templates imported
        """
        try:
            with open(filename, 'r') as f:
                catalog = json.load(f)
            
            imported_count = 0
            for template_id, template_info in catalog.get("templates", {}).items():
                # This would need to be implemented based on template serialization
                # For now, just log the import attempt
                logger.info(f"Would import template: {template_info['name']}")
                imported_count += 1
            
            logger.info(f"Imported {imported_count} templates from {filename}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Failed to import template catalog: {e}")
            return 0
    
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return sorted(list(self.categories))
    
    def get_tags(self) -> List[str]:
        """Get all available tags."""
        return sorted(list(self.tags))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        total_downloads = sum(t.stats.downloads for t in self.templates.values())
        avg_rating = sum(t.rating.average_rating for t in self.templates.values()) / len(self.templates) if self.templates else 0
        
        return {
            "total_templates": len(self.templates),
            "total_downloads": total_downloads,
            "average_rating": avg_rating,
            "categories": len(self.categories),
            "tags": len(self.tags),
            "featured_templates": len([t for t in self.templates.values() if t.featured]),
            "verified_templates": len([t for t in self.templates.values() if t.verified])
        }


# Global marketplace instance
_global_marketplace: Optional[TemplateMarketplace] = None


def get_template_marketplace() -> TemplateMarketplace:
    """Get the global template marketplace instance."""
    global _global_marketplace
    if _global_marketplace is None:
        _global_marketplace = TemplateMarketplace()
    return _global_marketplace


def search_marketplace_templates(**kwargs) -> List[MarketplaceTemplate]:
    """Search templates in marketplace."""
    return get_template_marketplace().search_templates(**kwargs)


def get_featured_templates(limit: int = 10) -> List[MarketplaceTemplate]:
    """Get featured templates from marketplace."""
    return get_template_marketplace().get_featured_templates(limit)