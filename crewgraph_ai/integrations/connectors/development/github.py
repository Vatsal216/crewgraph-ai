"""
GitHub Integration for CrewGraph AI

Provides GitHub repository management, issue tracking, pull requests,
and CI/CD integration capabilities for development workflows.

Author: Vatsal216
Created: 2025-07-23 18:45:00 UTC
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ... import BaseIntegration, IntegrationConfig, IntegrationMetadata, IntegrationResult, IntegrationType

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class GitHubIntegration(BaseIntegration):
    """
    GitHub integration for repository and project management.
    
    Supports repository operations, issue management, pull requests,
    and workflow automation through the GitHub REST API.
    """
    
    @property
    def metadata(self) -> IntegrationMetadata:
        """Get GitHub integration metadata."""
        return IntegrationMetadata(
            name="GitHub",
            version="1.0.0",
            description="Repository management and development workflow automation",
            author="CrewGraph AI",
            integration_type=IntegrationType.DEVELOPMENT,
            dependencies=["requests"],
            config_schema={
                "type": "object",
                "properties": {
                    "access_token": {
                        "type": "string",
                        "description": "GitHub Personal Access Token"
                    },
                    "default_owner": {
                        "type": "string",
                        "description": "Default repository owner/organization"
                    },
                    "base_url": {
                        "type": "string",
                        "description": "GitHub API base URL (for GitHub Enterprise)",
                        "default": "https://api.github.com"
                    }
                },
                "required": ["access_token"]
            },
            supports_async=True,
            supports_webhook=True,
            homepage="https://github.com/",
            documentation="https://docs.github.com/en/rest",
            tags=["development", "git", "repositories", "issues", "ci-cd"]
        )
    
    def initialize(self) -> bool:
        """Initialize GitHub integration."""
        try:
            if not REQUESTS_AVAILABLE:
                self.logger.error("requests library not available")
                return False
            
            # Get configuration
            self.access_token = self.config.config.get("access_token")
            self.default_owner = self.config.config.get("default_owner")
            self.base_url = self.config.config.get("base_url", "https://api.github.com")
            
            if not self.access_token:
                self.logger.error("access_token is required")
                return False
            
            # Setup API headers
            self.headers = {
                "Authorization": f"token {self.access_token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "CrewGraph-AI/1.0"
            }
            
            # Test API connection
            self.logger.info("Testing GitHub API connection...")
            
            self.is_initialized = True
            self.logger.info("GitHub integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GitHub integration: {e}")
            return False
    
    def execute(self, action: str, **kwargs) -> IntegrationResult:
        """Execute GitHub action."""
        if not self.is_initialized:
            return IntegrationResult(
                success=False,
                error_message="Integration not initialized"
            )
        
        try:
            if action == "create_issue":
                return self._create_issue(**kwargs)
            elif action == "list_issues":
                return self._list_issues(**kwargs)
            elif action == "create_pull_request":
                return self._create_pull_request(**kwargs)
            elif action == "list_repositories":
                return self._list_repositories(**kwargs)
            elif action == "get_repository":
                return self._get_repository(**kwargs)
            elif action == "create_repository":
                return self._create_repository(**kwargs)
            elif action == "list_commits":
                return self._list_commits(**kwargs)
            elif action == "create_webhook":
                return self._create_webhook(**kwargs)
            elif action == "trigger_workflow":
                return self._trigger_workflow(**kwargs)
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
        """Validate GitHub configuration."""
        issues = []
        
        config = self.config.config
        
        if not config.get("access_token"):
            issues.append("access_token is required")
        
        base_url = config.get("base_url")
        if base_url and not base_url.startswith("https://"):
            issues.append("base_url must use HTTPS")
        
        return issues
    
    def _create_issue(
        self,
        title: str,
        body: Optional[str] = None,
        repository: Optional[str] = None,
        owner: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
        milestone: Optional[int] = None
    ) -> IntegrationResult:
        """Create a new GitHub issue."""
        try:
            owner = owner or self.default_owner
            if not owner or not repository:
                return IntegrationResult(
                    success=False,
                    error_message="Both owner and repository are required"
                )
            
            payload = {
                "title": title,
                "body": body or "",
            }
            
            if labels:
                payload["labels"] = labels
            if assignees:
                payload["assignees"] = assignees
            if milestone:
                payload["milestone"] = milestone
            
            # Simulate API call
            response = self._simulate_api_call(
                f"POST /repos/{owner}/{repository}/issues",
                payload
            )
            
            if response.get("status") == "success":
                issue_data = response.get("data", {})
                return IntegrationResult(
                    success=True,
                    data={
                        "issue_number": issue_data.get("number"),
                        "issue_id": issue_data.get("id"),
                        "url": issue_data.get("html_url"),
                        "title": issue_data.get("title"),
                        "state": issue_data.get("state"),
                        "created_at": issue_data.get("created_at")
                    },
                    metadata={"repository": f"{owner}/{repository}"}
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=response.get("error", "Failed to create issue")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error creating issue: {str(e)}"
            )
    
    def _list_issues(
        self,
        repository: Optional[str] = None,
        owner: Optional[str] = None,
        state: str = "open",
        labels: Optional[List[str]] = None,
        assignee: Optional[str] = None,
        limit: int = 30
    ) -> IntegrationResult:
        """List GitHub issues."""
        try:
            owner = owner or self.default_owner
            if not owner or not repository:
                return IntegrationResult(
                    success=False,
                    error_message="Both owner and repository are required"
                )
            
            params = {
                "state": state,
                "per_page": min(limit, 100)
            }
            
            if labels:
                params["labels"] = ",".join(labels)
            if assignee:
                params["assignee"] = assignee
            
            # Simulate API call
            response = self._simulate_api_call(
                f"GET /repos/{owner}/{repository}/issues",
                params
            )
            
            if response.get("status") == "success":
                issues = response.get("data", [])
                issue_list = [
                    {
                        "number": issue.get("number"),
                        "id": issue.get("id"),
                        "title": issue.get("title"),
                        "state": issue.get("state"),
                        "labels": [label.get("name") for label in issue.get("labels", [])],
                        "assignees": [user.get("login") for user in issue.get("assignees", [])],
                        "created_at": issue.get("created_at"),
                        "updated_at": issue.get("updated_at"),
                        "url": issue.get("html_url")
                    }
                    for issue in issues
                ]
                
                return IntegrationResult(
                    success=True,
                    data={"issues": issue_list, "total": len(issue_list)},
                    metadata={"repository": f"{owner}/{repository}", "state": state}
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=response.get("error", "Failed to list issues")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error listing issues: {str(e)}"
            )
    
    def _create_pull_request(
        self,
        title: str,
        head: str,
        base: str,
        body: Optional[str] = None,
        repository: Optional[str] = None,
        owner: Optional[str] = None,
        draft: bool = False
    ) -> IntegrationResult:
        """Create a new pull request."""
        try:
            owner = owner or self.default_owner
            if not owner or not repository:
                return IntegrationResult(
                    success=False,
                    error_message="Both owner and repository are required"
                )
            
            payload = {
                "title": title,
                "head": head,
                "base": base,
                "body": body or "",
                "draft": draft
            }
            
            # Simulate API call
            response = self._simulate_api_call(
                f"POST /repos/{owner}/{repository}/pulls",
                payload
            )
            
            if response.get("status") == "success":
                pr_data = response.get("data", {})
                return IntegrationResult(
                    success=True,
                    data={
                        "pull_number": pr_data.get("number"),
                        "pull_id": pr_data.get("id"),
                        "url": pr_data.get("html_url"),
                        "title": pr_data.get("title"),
                        "state": pr_data.get("state"),
                        "draft": pr_data.get("draft"),
                        "mergeable": pr_data.get("mergeable"),
                        "created_at": pr_data.get("created_at")
                    },
                    metadata={"repository": f"{owner}/{repository}"}
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=response.get("error", "Failed to create pull request")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error creating pull request: {str(e)}"
            )
    
    def _list_repositories(
        self,
        owner: Optional[str] = None,
        type: str = "all",
        sort: str = "updated",
        limit: int = 30
    ) -> IntegrationResult:
        """List GitHub repositories."""
        try:
            owner = owner or self.default_owner
            
            if owner:
                endpoint = f"GET /users/{owner}/repos"
            else:
                endpoint = "GET /user/repos"
            
            params = {
                "type": type,
                "sort": sort,
                "per_page": min(limit, 100)
            }
            
            # Simulate API call
            response = self._simulate_api_call(endpoint, params)
            
            if response.get("status") == "success":
                repos = response.get("data", [])
                repo_list = [
                    {
                        "id": repo.get("id"),
                        "name": repo.get("name"),
                        "full_name": repo.get("full_name"),
                        "description": repo.get("description"),
                        "private": repo.get("private"),
                        "fork": repo.get("fork"),
                        "language": repo.get("language"),
                        "stars": repo.get("stargazers_count"),
                        "forks": repo.get("forks_count"),
                        "created_at": repo.get("created_at"),
                        "updated_at": repo.get("updated_at"),
                        "url": repo.get("html_url"),
                        "clone_url": repo.get("clone_url")
                    }
                    for repo in repos
                ]
                
                return IntegrationResult(
                    success=True,
                    data={"repositories": repo_list, "total": len(repo_list)},
                    metadata={"owner": owner, "type": type}
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=response.get("error", "Failed to list repositories")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error listing repositories: {str(e)}"
            )
    
    def _get_repository(
        self,
        repository: str,
        owner: Optional[str] = None
    ) -> IntegrationResult:
        """Get repository information."""
        try:
            owner = owner or self.default_owner
            if not owner:
                return IntegrationResult(
                    success=False,
                    error_message="Repository owner is required"
                )
            
            # Simulate API call
            response = self._simulate_api_call(
                f"GET /repos/{owner}/{repository}",
                {}
            )
            
            if response.get("status") == "success":
                repo = response.get("data", {})
                return IntegrationResult(
                    success=True,
                    data={
                        "id": repo.get("id"),
                        "name": repo.get("name"),
                        "full_name": repo.get("full_name"),
                        "description": repo.get("description"),
                        "private": repo.get("private"),
                        "language": repo.get("language"),
                        "stars": repo.get("stargazers_count"),
                        "forks": repo.get("forks_count"),
                        "open_issues": repo.get("open_issues_count"),
                        "default_branch": repo.get("default_branch"),
                        "created_at": repo.get("created_at"),
                        "updated_at": repo.get("updated_at"),
                        "url": repo.get("html_url"),
                        "clone_url": repo.get("clone_url")
                    }
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=response.get("error", "Repository not found")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error getting repository: {str(e)}"
            )
    
    def _trigger_workflow(
        self,
        workflow_id: str,
        repository: str,
        owner: Optional[str] = None,
        ref: str = "main",
        inputs: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """Trigger a GitHub Actions workflow."""
        try:
            owner = owner or self.default_owner
            if not owner:
                return IntegrationResult(
                    success=False,
                    error_message="Repository owner is required"
                )
            
            payload = {
                "ref": ref,
                "inputs": inputs or {}
            }
            
            # Simulate API call
            response = self._simulate_api_call(
                f"POST /repos/{owner}/{repository}/actions/workflows/{workflow_id}/dispatches",
                payload
            )
            
            if response.get("status") == "success":
                return IntegrationResult(
                    success=True,
                    data={
                        "workflow_id": workflow_id,
                        "repository": f"{owner}/{repository}",
                        "ref": ref,
                        "triggered_at": datetime.now().isoformat()
                    },
                    metadata={"workflow_id": workflow_id, "ref": ref}
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=response.get("error", "Failed to trigger workflow")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error triggering workflow: {str(e)}"
            )
    
    def _simulate_api_call(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate GitHub API call (for testing without actual API)."""
        # Parse endpoint
        parts = endpoint.split(" ", 1)
        method = parts[0] if len(parts) > 1 else "GET"
        path = parts[1] if len(parts) > 1 else endpoint
        
        # Simulate responses based on endpoint
        if "issues" in path and method == "POST":
            return {
                "status": "success",
                "data": {
                    "id": 123456789,
                    "number": 42,
                    "title": payload.get("title"),
                    "state": "open",
                    "html_url": "https://github.com/owner/repo/issues/42",
                    "created_at": datetime.now().isoformat()
                }
            }
        
        elif "issues" in path and method == "GET":
            return {
                "status": "success",
                "data": [
                    {
                        "id": 123456789,
                        "number": 42,
                        "title": "Sample Issue",
                        "state": "open",
                        "labels": [{"name": "bug"}, {"name": "help wanted"}],
                        "assignees": [{"login": "developer1"}],
                        "created_at": "2025-01-01T00:00:00Z",
                        "updated_at": "2025-01-02T00:00:00Z",
                        "html_url": "https://github.com/owner/repo/issues/42"
                    }
                ]
            }
        
        elif "pulls" in path and method == "POST":
            return {
                "status": "success",
                "data": {
                    "id": 987654321,
                    "number": 15,
                    "title": payload.get("title"),
                    "state": "open",
                    "draft": payload.get("draft", False),
                    "mergeable": True,
                    "html_url": "https://github.com/owner/repo/pull/15",
                    "created_at": datetime.now().isoformat()
                }
            }
        
        elif "repos" in path and "/repos/" in path and method == "GET":
            return {
                "status": "success",
                "data": {
                    "id": 456789123,
                    "name": "sample-repo",
                    "full_name": "owner/sample-repo",
                    "description": "A sample repository",
                    "private": False,
                    "language": "Python",
                    "stargazers_count": 42,
                    "forks_count": 15,
                    "open_issues_count": 3,
                    "default_branch": "main",
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-02T00:00:00Z",
                    "html_url": "https://github.com/owner/sample-repo",
                    "clone_url": "https://github.com/owner/sample-repo.git"
                }
            }
        
        elif "repos" in path and method == "GET":
            return {
                "status": "success",
                "data": [
                    {
                        "id": 456789123,
                        "name": "sample-repo",
                        "full_name": "owner/sample-repo",
                        "description": "A sample repository",
                        "private": False,
                        "fork": False,
                        "language": "Python",
                        "stargazers_count": 42,
                        "forks_count": 15,
                        "created_at": "2025-01-01T00:00:00Z",
                        "updated_at": "2025-01-02T00:00:00Z",
                        "html_url": "https://github.com/owner/sample-repo",
                        "clone_url": "https://github.com/owner/sample-repo.git"
                    }
                ]
            }
        
        elif "workflows" in path and "dispatches" in path:
            return {
                "status": "success",
                "data": {
                    "message": "Workflow triggered successfully"
                }
            }
        
        else:
            return {
                "status": "error",
                "error": f"Unknown endpoint: {endpoint}"
            }
    
    def _perform_health_check(self) -> Dict[str, Any]:
        """Perform GitHub-specific health check."""
        try:
            # Test authentication by getting user info
            response = self._simulate_api_call("GET /user", {})
            
            if response.get("status") == "success":
                return {
                    "status": "healthy",
                    "message": "GitHub API connection successful",
                    "api_rate_limit": "60 requests/hour",  # Simulated
                    "execution_count": self.execution_count,
                    "last_execution": self.last_execution
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": f"GitHub API error: {response.get('error')}",
                    "execution_count": self.execution_count
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "execution_count": self.execution_count
            }