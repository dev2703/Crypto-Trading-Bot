"""
Rate Limiter Module

This module provides rate limiting functionality for API requests.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int = 50):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}
        
    def _clean_old_requests(self, endpoint: str):
        """Remove requests older than 1 minute."""
        now = datetime.now()
        self.requests[endpoint] = [
            req_time for req_time in self.requests.get(endpoint, [])
            if now - req_time < timedelta(minutes=1)
        ]
        
    def wait_if_needed(self, endpoint: str):
        """Wait if rate limit is reached."""
        if endpoint not in self.requests:
            self.requests[endpoint] = []
            
        self._clean_old_requests(endpoint)
        
        if len(self.requests[endpoint]) >= self.requests_per_minute:
            oldest_request = self.requests[endpoint][0]
            wait_time = 60 - (datetime.now() - oldest_request).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached for {endpoint}, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
            self._clean_old_requests(endpoint)
            
        self.requests[endpoint].append(datetime.now())
        
class Cache:
    """Simple in-memory cache for API responses."""
    
    def __init__(self, ttl_seconds: int = 300):  # 5 minutes default TTL
        self.cache: Dict[str, Dict] = {}
        self.ttl_seconds = ttl_seconds
        
    def get(self, key: str) -> Optional[Dict]:
        """Get cached value if not expired."""
        if key in self.cache:
            cached_data = self.cache[key]
            if datetime.now().timestamp() - cached_data['timestamp'] < self.ttl_seconds:
                return cached_data['data']
            else:
                del self.cache[key]
        return None
        
    def set(self, key: str, value: Dict):
        """Set cache value with timestamp."""
        self.cache[key] = {
            'data': value,
            'timestamp': datetime.now().timestamp()
        }
        
    def clear(self):
        """Clear all cached data."""
        self.cache.clear() 