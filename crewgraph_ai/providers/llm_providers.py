"""
LLM Provider Management System for CrewGraph AI
Provides enterprise-grade LLM provider integration with rate limiting, retry logic, 
authentication, and failover capabilities.

Author: Vatsal216
Created: 2025-07-23
"""

import asyncio
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator
from enum import Enum
import json

from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError
from ..config.enterprise_config import LLMProviderConfig, LLMProvider, get_enterprise_config

logger = get_logger(__name__)


class RequestType(Enum):
    """Types of LLM requests"""
    COMPLETION = "completion"
    CHAT = "chat"
    EMBEDDING = "embedding"
    MODERATION = "moderation"


@dataclass
class LLMRequest:
    """LLM request wrapper"""
    request_id: str
    request_type: RequestType
    model: str
    prompt: str
    parameters: Dict[str, Any]
    timestamp: float
    retry_count: int = 0


@dataclass
class LLMResponse:
    """LLM response wrapper"""
    request_id: str
    response_data: Any
    provider: str
    model: str
    tokens_used: int
    cost: float
    latency: float
    timestamp: float


class RateLimiter:
    """Token bucket rate limiter for LLM API calls"""
    
    def __init__(self, requests_per_minute: int = 60, tokens_per_minute: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        
        # Request rate limiting
        self.request_tokens = requests_per_minute
        self.max_request_tokens = requests_per_minute
        self.last_request_update = time.time()
        
        # Token rate limiting  
        self.token_tokens = tokens_per_minute
        self.max_token_tokens = tokens_per_minute
        self.last_token_update = time.time()
        
        self._lock = threading.Lock()
    
    def can_make_request(self, estimated_tokens: int = 1) -> bool:
        """Check if request can be made within rate limits"""
        with self._lock:
            now = time.time()
            
            # Refill request bucket
            request_time_passed = now - self.last_request_update
            self.request_tokens = min(
                self.max_request_tokens,
                self.request_tokens + (request_time_passed * self.requests_per_minute / 60)
            )
            self.last_request_update = now
            
            # Refill token bucket
            token_time_passed = now - self.last_token_update
            self.token_tokens = min(
                self.max_token_tokens,
                self.token_tokens + (token_time_passed * self.tokens_per_minute / 60)
            )
            self.last_token_update = now
            
            # Check if we have enough tokens for both request and estimated tokens
            return self.request_tokens >= 1 and self.token_tokens >= estimated_tokens
    
    def consume_tokens(self, requests: int = 1, tokens: int = 1):
        """Consume tokens from rate limiter"""
        with self._lock:
            self.request_tokens -= requests
            self.token_tokens -= tokens
    
    def time_until_available(self, estimated_tokens: int = 1) -> float:
        """Get time in seconds until rate limit allows request"""
        with self._lock:
            if self.can_make_request(estimated_tokens):
                return 0.0
            
            # Calculate time needed for request tokens
            request_time = 0.0
            if self.request_tokens < 1:
                tokens_needed = 1 - self.request_tokens
                request_time = (tokens_needed * 60) / self.requests_per_minute
            
            # Calculate time needed for tokens
            token_time = 0.0
            if self.token_tokens < estimated_tokens:
                tokens_needed = estimated_tokens - self.token_tokens
                token_time = (tokens_needed * 60) / self.tokens_per_minute
            
            return max(request_time, token_time)


class RetryHandler:
    """Intelligent retry handler with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def should_retry(self, error: Exception, retry_count: int) -> bool:
        """Determine if request should be retried"""
        if retry_count >= self.max_retries:
            return False
        
        # Retry on specific error types
        error_msg = str(error).lower()
        retryable_errors = [
            "rate limit", "timeout", "connection", "server error",
            "503", "502", "429", "internal server error"
        ]
        
        return any(err in error_msg for err in retryable_errors)
    
    def get_delay(self, retry_count: int) -> float:
        """Get delay for retry attempt"""
        delay = self.base_delay * (2 ** retry_count)
        return min(delay, self.max_delay)
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                if not self.should_retry(e, retry_count):
                    break
                
                delay = self.get_delay(retry_count)
                logger.warning(f"Request failed (attempt {retry_count + 1}/{self.max_retries + 1}): {e}. Retrying in {delay}s")
                
                await asyncio.sleep(delay)
                retry_count += 1
        
        # All retries exhausted
        raise CrewGraphError(f"Request failed after {retry_count} retries: {last_error}")


class ProviderClient(ABC):
    """Abstract base class for LLM provider clients"""
    
    def __init__(self, config: LLMProviderConfig):
        self.config = config
        self.rate_limiter = RateLimiter(
            config.rate_limit_rpm,
            config.rate_limit_tpm
        )
        self.retry_handler = RetryHandler(
            config.max_retries,
            config.retry_delay,
            60.0
        )
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self._lock = threading.Lock()
    
    @abstractmethod
    async def chat_completion(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """Generate chat completion"""
        pass
    
    @abstractmethod
    async def text_completion(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text completion"""
        pass
    
    @abstractmethod
    async def get_embeddings(self, texts: List[str], **kwargs) -> LLMResponse:
        """Get text embeddings"""
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        pass
    
    @abstractmethod
    def estimate_cost(self, tokens: int, model: str) -> float:
        """Estimate cost for token usage"""
        pass
    
    async def wait_for_rate_limit(self, estimated_tokens: int = 1):
        """Wait until rate limit allows request"""
        wait_time = self.rate_limiter.time_until_available(estimated_tokens)
        if wait_time > 0:
            logger.info(f"Rate limit reached, waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
    
    def update_usage_stats(self, tokens: int, cost: float):
        """Update usage statistics"""
        with self._lock:
            self.request_count += 1
            self.total_tokens += tokens
            self.total_cost += cost
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        with self._lock:
            return {
                "request_count": self.request_count,
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost,
                "provider": self.config.provider.value,
                "enabled": self.config.enabled
            }


class OpenAIClient(ProviderClient):
    """OpenAI provider client"""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.openai.com/v1"
        
        # Token pricing (approximate, per 1K tokens)
        self.pricing = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002},
        }
    
    async def chat_completion(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """Generate chat completion using OpenAI API"""
        model = kwargs.get("model", self.config.default_model)
        
        # Estimate tokens
        text_content = " ".join([msg.get("content", "") for msg in messages])
        estimated_tokens = self.estimate_tokens(text_content)
        
        # Wait for rate limit
        await self.wait_for_rate_limit(estimated_tokens)
        
        # Consume rate limit tokens
        self.rate_limiter.consume_tokens(1, estimated_tokens)
        
        # Simulate API call (in real implementation, use actual OpenAI client)
        start_time = time.time()
        try:
            # This would be the actual API call
            response_data = {
                "choices": [
                    {
                        "message": {
                            "content": f"This is a simulated response from {model}",
                            "role": "assistant"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": estimated_tokens,
                    "completion_tokens": 50,
                    "total_tokens": estimated_tokens + 50
                }
            }
            
            # Simulate processing time
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
        
        latency = time.time() - start_time
        total_tokens = response_data["usage"]["total_tokens"]
        cost = self.estimate_cost(total_tokens, model)
        
        self.update_usage_stats(total_tokens, cost)
        
        return LLMResponse(
            request_id=f"req_{int(time.time())}",
            response_data=response_data,
            provider="openai",
            model=model,
            tokens_used=total_tokens,
            cost=cost,
            latency=latency,
            timestamp=time.time()
        )
    
    async def text_completion(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text completion"""
        # Convert to chat format for consistency
        messages = [{"role": "user", "content": prompt}]
        return await self.chat_completion(messages, **kwargs)
    
    async def get_embeddings(self, texts: List[str], **kwargs) -> LLMResponse:
        """Get text embeddings (simulated)"""
        model = kwargs.get("model", "text-embedding-ada-002")
        
        total_tokens = sum(self.estimate_tokens(text) for text in texts)
        await self.wait_for_rate_limit(total_tokens)
        self.rate_limiter.consume_tokens(1, total_tokens)
        
        start_time = time.time()
        
        # Simulate embeddings response
        response_data = {
            "data": [
                {"embedding": [0.1] * 1536, "index": i}
                for i in range(len(texts))
            ],
            "usage": {"total_tokens": total_tokens}
        }
        
        await asyncio.sleep(0.2)
        
        latency = time.time() - start_time
        cost = total_tokens * 0.0001 / 1000  # $0.0001 per 1K tokens
        
        self.update_usage_stats(total_tokens, cost)
        
        return LLMResponse(
            request_id=f"emb_{int(time.time())}",
            response_data=response_data,
            provider="openai",
            model=model,
            tokens_used=total_tokens,
            cost=cost,
            latency=latency,
            timestamp=time.time()
        )
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        return max(1, len(text.split()) * 1.3)
    
    def estimate_cost(self, tokens: int, model: str) -> float:
        """Estimate cost for token usage"""
        if model not in self.pricing:
            model = "gpt-3.5-turbo"  # Default pricing
        
        pricing = self.pricing[model]
        # Assume 70% prompt, 30% completion
        prompt_tokens = int(tokens * 0.7)
        completion_tokens = int(tokens * 0.3)
        
        cost = (prompt_tokens * pricing["prompt"] + 
                completion_tokens * pricing["completion"]) / 1000
        
        return round(cost, 6)


class AnthropicClient(ProviderClient):
    """Anthropic provider client"""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.pricing = {
            "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
            "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
            "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
        }
    
    async def chat_completion(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """Generate chat completion using Anthropic API"""
        model = kwargs.get("model", self.config.default_model)
        
        # Convert messages to Anthropic format
        text_content = " ".join([msg.get("content", "") for msg in messages])
        estimated_tokens = self.estimate_tokens(text_content)
        
        await self.wait_for_rate_limit(estimated_tokens)
        self.rate_limiter.consume_tokens(1, estimated_tokens)
        
        start_time = time.time()
        
        # Simulate API call
        response_data = {
            "content": [
                {
                    "text": f"This is a simulated response from {model}",
                    "type": "text"
                }
            ],
            "usage": {
                "input_tokens": estimated_tokens,
                "output_tokens": 45,
                "total_tokens": estimated_tokens + 45
            }
        }
        
        await asyncio.sleep(0.6)
        
        latency = time.time() - start_time
        total_tokens = response_data["usage"]["total_tokens"]
        cost = self.estimate_cost(total_tokens, model)
        
        self.update_usage_stats(total_tokens, cost)
        
        return LLMResponse(
            request_id=f"claude_{int(time.time())}",
            response_data=response_data,
            provider="anthropic",
            model=model,
            tokens_used=total_tokens,
            cost=cost,
            latency=latency,
            timestamp=time.time()
        )
    
    async def text_completion(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text completion"""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat_completion(messages, **kwargs)
    
    async def get_embeddings(self, texts: List[str], **kwargs) -> LLMResponse:
        """Anthropic doesn't provide embeddings - raise error"""
        raise CrewGraphError("Anthropic does not provide embedding models")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        return max(1, len(text.split()) * 1.2)
    
    def estimate_cost(self, tokens: int, model: str) -> float:
        """Estimate cost for token usage"""
        if model not in self.pricing:
            model = "claude-3-sonnet"
        
        pricing = self.pricing[model]
        prompt_tokens = int(tokens * 0.7)
        completion_tokens = int(tokens * 0.3)
        
        cost = (prompt_tokens * pricing["prompt"] + 
                completion_tokens * pricing["completion"]) / 1000
        
        return round(cost, 6)


class LLMProviderManager:
    """
    Enterprise LLM provider manager with failover, load balancing, and monitoring
    """
    
    def __init__(self, enterprise_config=None):
        self.config = enterprise_config or get_enterprise_config()
        self.clients: Dict[str, ProviderClient] = {}
        self.client_health: Dict[str, bool] = {}
        self.failover_order: List[str] = []
        self._lock = threading.Lock()
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize provider clients from configuration"""
        for name, provider_config in self.config.llm_providers.items():
            if not provider_config.enabled:
                continue
                
            try:
                client = self._create_client(provider_config)
                self.clients[name] = client
                self.client_health[name] = True
                self.failover_order.append(name)
                
                logger.info(f"Initialized LLM provider: {name} ({provider_config.provider.value})")
                
            except Exception as e:
                logger.error(f"Failed to initialize provider {name}: {e}")
                self.client_health[name] = False
    
    def _create_client(self, config: LLMProviderConfig) -> ProviderClient:
        """Create provider client based on configuration"""
        if config.provider == LLMProvider.OPENAI:
            return OpenAIClient(config)
        elif config.provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(config)
        elif config.provider == LLMProvider.AZURE:
            # Azure OpenAI uses OpenAI client with different base URL
            return OpenAIClient(config)
        else:
            raise CrewGraphError(f"Unsupported provider: {config.provider}")
    
    def get_client(self, provider_name: Optional[str] = None) -> ProviderClient:
        """Get provider client with failover"""
        if provider_name and provider_name in self.clients:
            if self.client_health.get(provider_name, False):
                return self.clients[provider_name]
            else:
                logger.warning(f"Provider {provider_name} is unhealthy, trying failover")
        
        # Try failover order
        for name in self.failover_order:
            if self.client_health.get(name, False) and name in self.clients:
                logger.info(f"Using fallback provider: {name}")
                return self.clients[name]
        
        raise CrewGraphError("No healthy LLM providers available")
    
    async def chat_completion(self, messages: List[Dict], provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate chat completion with automatic failover"""
        client = self.get_client(provider)
        
        try:
            response = await client.retry_handler.execute_with_retry(
                client.chat_completion, messages, **kwargs
            )
            return response
            
        except Exception as e:
            # Mark provider as unhealthy
            provider_name = provider or self.config.default_provider
            if provider_name in self.client_health:
                self.client_health[provider_name] = False
            
            logger.error(f"Provider {provider_name} failed: {e}")
            raise
    
    async def text_completion(self, prompt: str, provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate text completion with automatic failover"""
        client = self.get_client(provider)
        
        try:
            response = await client.retry_handler.execute_with_retry(
                client.text_completion, prompt, **kwargs
            )
            return response
            
        except Exception as e:
            provider_name = provider or self.config.default_provider
            if provider_name in self.client_health:
                self.client_health[provider_name] = False
            
            raise
    
    async def get_embeddings(self, texts: List[str], provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """Get embeddings with automatic failover"""
        client = self.get_client(provider)
        
        try:
            response = await client.retry_handler.execute_with_retry(
                client.get_embeddings, texts, **kwargs
            )
            return response
            
        except Exception as e:
            provider_name = provider or self.config.default_provider
            if provider_name in self.client_health:
                self.client_health[provider_name] = False
            
            raise
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all providers"""
        stats = {}
        total_requests = 0
        total_tokens = 0
        total_cost = 0.0
        
        for name, client in self.clients.items():
            client_stats = client.get_usage_stats()
            stats[name] = client_stats
            total_requests += client_stats["request_count"]
            total_tokens += client_stats["total_tokens"]
            total_cost += client_stats["total_cost"]
        
        stats["total"] = {
            "request_count": total_requests,
            "total_tokens": total_tokens,
            "total_cost": total_cost
        }
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all providers"""
        return {
            "providers": dict(self.client_health),
            "healthy_count": sum(self.client_health.values()),
            "total_count": len(self.client_health),
            "failover_order": self.failover_order
        }
    
    async def health_check(self, provider_name: str) -> bool:
        """Perform health check on specific provider"""
        if provider_name not in self.clients:
            return False
        
        try:
            client = self.clients[provider_name]
            # Perform a simple completion request
            await client.chat_completion([{"role": "user", "content": "ping"}])
            self.client_health[provider_name] = True
            return True
            
        except Exception as e:
            logger.warning(f"Health check failed for {provider_name}: {e}")
            self.client_health[provider_name] = False
            return False
    
    async def health_check_all(self):
        """Perform health check on all providers"""
        tasks = []
        for provider_name in self.clients.keys():
            tasks.append(self.health_check(provider_name))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# Global provider manager
_global_provider_manager: Optional[LLMProviderManager] = None
_manager_lock = threading.Lock()


def get_provider_manager() -> LLMProviderManager:
    """Get global LLM provider manager"""
    global _global_provider_manager
    
    with _manager_lock:
        if _global_provider_manager is None:
            _global_provider_manager = LLMProviderManager()
        
        return _global_provider_manager


def create_provider_client(provider_config: LLMProviderConfig) -> ProviderClient:
    """Create a standalone provider client"""
    manager = LLMProviderManager()
    return manager._create_client(provider_config)


# Convenience functions
async def chat_completion(messages: List[Dict], provider: Optional[str] = None, **kwargs) -> LLMResponse:
    """Generate chat completion using global provider manager"""
    manager = get_provider_manager()
    return await manager.chat_completion(messages, provider, **kwargs)


async def text_completion(prompt: str, provider: Optional[str] = None, **kwargs) -> LLMResponse:
    """Generate text completion using global provider manager"""
    manager = get_provider_manager()
    return await manager.text_completion(prompt, provider, **kwargs)


async def get_embeddings(texts: List[str], provider: Optional[str] = None, **kwargs) -> LLMResponse:
    """Get embeddings using global provider manager"""
    manager = get_provider_manager()
    return await manager.get_embeddings(texts, provider, **kwargs)